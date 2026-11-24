import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# 1. 读取节点特征
# ======================
def load_node_features(path):
    df = pd.read_csv(path, header=None)
    ids = df.iloc[:, 0].astype(str).values
    feats = df.iloc[:, 1:].values.astype(np.float32)

    id2idx = {id_: i for i, id_ in enumerate(ids)}
    return ids, feats, id2idx


# ======================
# 2. 构建邻接矩阵 + edge_index
# ======================
def build_adj_and_edges(dti_path, id2idx):
    df = pd.read_csv(dti_path)

    N = len(id2idx)
    edges = []

    for d, t in zip(df["drug"], df["target"]):
        d, t = str(d), str(t)
        if d in id2idx and t in id2idx:
            i = id2idx[d]
            j = id2idx[t]
            edges.append((i, j))
            edges.append((j, i))

    if len(edges) == 0:
        raise ValueError("No valid edges found in dti.csv.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    values = torch.ones(edge_index.shape[1], dtype=torch.float32)

    adj = torch.sparse_coo_tensor(edge_index, values, (N, N)).coalesce()
    return adj, edge_index


# ======================
# 3. diffusion
# ======================
def fast_diffusion(adj, x, alpha):
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv = 1.0 / (deg + 1e-6)

    x_prop = torch.sparse.mm(adj, x)
    x_prop = deg_inv.unsqueeze(1) * x_prop

    return (1 - alpha) * x + alpha * x_prop


def multi_scale_diffusion(adj, x, scales=None):
    if scales is None:
        scales = [0.1, 0.3, 0.6]

    outs = []
    for alpha in scales:
        outs.append(fast_diffusion(adj, x, alpha))
    return outs


# ======================
# 4. 数据增强
# ======================
def augment_features(x, feat_drop=0.1, noise_std=0.01):
    if feat_drop > 0:
        mask = (torch.rand_like(x) > feat_drop).float()
        x = x * mask
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x


# ======================
# 5. 多尺度融合
# ======================
class MultiScaleFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_scales):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_scales)
        ])
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, diff_feats):
        hs = []
        for i, feat in enumerate(diff_feats):
            hs.append(F.relu(self.proj[i](feat)))  # [N, H]

        h_stack = torch.stack(hs, dim=1)          # [N, S, H]
        score = self.att(h_stack)                 # [N, S, 1]
        weight = torch.softmax(score, dim=1)      # [N, S, 1]
        h = (h_stack * weight).sum(dim=1)         # [N, H]
        return h


# ======================
# 6. Graph Transformer Layer
# ======================
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index):
        """
        x: [N, H]
        edge_index: [2, E], src -> dst
        """
        edge_index = edge_index.to(x.device)   # ⭐ 关键修复
        N = x.size(0)
        src, dst = edge_index

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        q_dst = q[dst]   # [E, heads, dim]
        k_src = k[src]   # [E, heads, dim]
        v_src = v[src]   # [E, heads, dim]

        attn_score = (q_dst * k_src).sum(dim=-1) / self.scale
        attn_score = attn_score.clamp(-10, 10)

        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)

        for h in range(self.num_heads):
            score_h = attn_score[:, h]          # [E]
            exp_score = torch.exp(score_h)

            denom = torch.zeros(N, device=x.device)
            denom.index_add_(0, dst, exp_score)

            alpha = exp_score / (denom[dst] + 1e-9)   # [E]
            msg = v_src[:, h, :] * alpha.unsqueeze(-1)

            out[:, h, :].index_add_(0, dst, msg)

        out = out.reshape(N, self.hidden_dim)
        out = self.out_proj(out)

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.ffn(x))
        return x


# ======================
# 7. 整体模型
# ======================
class DiffusionGraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        num_scales=3
    ):
        super().__init__()

        self.fusion = MultiScaleFusion(input_dim, hidden_dim, num_scales)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, diff_feats, edge_index):
        x = self.fusion(diff_feats)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.out_norm(x)
        return x


# ======================
# 8. 对比学习损失
# ======================
def contrastive_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    sim = torch.matmul(z1, z2.T) / tau
    labels = torch.arange(z1.size(0), device=z1.device)

    loss1 = F.cross_entropy(sim, labels)
    loss2 = F.cross_entropy(sim.T, labels)
    return (loss1 + loss2) / 2


# ======================
# 9. 训练
# ======================
def train(
    adj,
    edge_index,
    feats,
    epochs=30,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    lr=1e-3,
    scales=None,
    feat_drop=0.1,
    noise_std=0.01
):
    feats = torch.tensor(feats, dtype=torch.float32).to(device)
    adj = adj.to(device)
    edge_index = edge_index.to(device)

    print("Computing diffusion...")
    base_diff_feats = multi_scale_diffusion(adj, feats, scales=scales)

    model = DiffusionGraphTransformer(
        input_dim=feats.shape[1],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        num_scales=len(base_diff_feats)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()

        diff_view1 = [augment_features(x.clone(), feat_drop, noise_std) for x in base_diff_feats]
        diff_view2 = [augment_features(x.clone(), feat_drop, noise_std) for x in base_diff_feats]

        z1 = model(diff_view1, edge_index)
        z2 = model(diff_view2, edge_index)

        loss = contrastive_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:03d}, Loss: {loss.item():.4f}")

    return model, base_diff_feats


# ======================
# 10. 获取 embedding
# ======================
@torch.no_grad()
def get_embeddings(model, diff_feats, edge_index):
    model.eval()
    edge_index = edge_index.to(device)              # ⭐ 再加一层保险
    diff_feats = [x.to(device) for x in diff_feats]
    z = model(diff_feats, edge_index)
    return z.cpu().numpy()


# ======================
# main
# ======================
if __name__ == "__main__":
    NODE_FILE = "../dataset/davis/node_features.csv"
    DTI_FILE = "../dataset/davis/dti.csv"

    ids, feats, id2idx = load_node_features(NODE_FILE)
    adj, edge_index = build_adj_and_edges(DTI_FILE, id2idx)

    model, diff_feats = train(
        adj,
        edge_index,
        feats,
        epochs=30,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        lr=1e-3,
        scales=[0.1, 0.3, 0.6],
        feat_drop=0.1,
        noise_std=0.01
    )

    emb = get_embeddings(model, diff_feats, edge_index)

    df = pd.concat([pd.DataFrame(ids), pd.DataFrame(emb)], axis=1)
    df.to_csv("node_embeddings.csv", index=False, header=False)

    print("✅ Done!")
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------- #

def seg_softmax(s: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    mx = torch.full((n,), -torch.inf, dtype=s.dtype, device=s.device)
    mx = mx.scatter_reduce(0, idx, s, reduce="amax", include_self=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    e = torch.exp(s - mx[idx])
    den = torch.zeros(n, dtype=s.dtype, device=s.device).index_add(0, idx, e)
    return e / (den[idx] + 1e-16)


def cos_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).mean()



# --------------------------------------------------------------------------- #
class Hypergraph:
    """Incidence lists (node i, hyperedge e) plus the degree inverses of Eq.(2)."""

    def __init__(self, node: np.ndarray, edge: np.ndarray, n_nodes: int, device):
        self.node = torch.from_numpy(node).to(device)
        self.edge = torch.from_numpy(edge).to(device)
        self.n_edges = int(edge.max()) + 1 if edge.size else 0
        self.n_nodes = n_nodes
        dv = np.bincount(node, minlength=n_nodes).astype(np.float32)
        de = np.bincount(edge, minlength=self.n_edges).astype(np.float32)
        self.dv_inv = torch.from_numpy(np.power(np.maximum(dv, 1), -0.5)).to(device)
        self.de_inv = torch.from_numpy(1.0 / np.maximum(de, 1.0)).to(device)


def build_hypergraph(x: np.ndarray, k: int = 5, device="cpu") -> Hypergraph:
    """kNN hypergraph: hyperedge e_i = {i} U top-k cosine neighbours of i.

    Rebuilt from the *current* node representations at every layer, as described
    in Section 3.1 ("each layer is followed by a kNN construction of a new H").
    """
    n = x.shape[0]
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    nn_ = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine").fit(xn)
    _, idx = nn_.kneighbors(xn)
    members = [np.concatenate(([v], idx[v, 1:])) for v in range(n)]
    node = np.concatenate(members)
    edge = np.concatenate([np.full(m.size, i) for i, m in enumerate(members)])
    return Hypergraph(node.astype(np.int64), edge.astype(np.int64), n, device)


# --------------------------------------------------------------------------- #
class HyperConv(nn.Module):
    """X' = sigma(Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta)           Eq.(4)"""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.theta = nn.Linear(d_in, d_out, bias=False)
        self.w = nn.Parameter(torch.ones(1))        # W, initialised to identity

    def forward(self, x, hg: Hypergraph):
        h = self.theta(x)
        # H^T Dv^-1/2 X, then De^-1 and W       
        msg = hg.dv_inv[hg.node].unsqueeze(-1) * h[hg.node]
        m = seg_sum(msg, hg.edge, hg.n_edges) * hg.de_inv.unsqueeze(-1) * self.w
        # H m, then Dv^-1/2 
        return seg_sum(m[hg.edge], hg.node, x.shape[0]) * hg.dv_inv.unsqueeze(-1)


class VertexConv(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, hg: Hypergraph):
        h = self.lin(x)
        return seg_sum(h[hg.node], hg.edge, hg.n_edges) * hg.de_inv.unsqueeze(-1)


class EdgeConv(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, m, hg: Hypergraph, n_nodes: int):
        return seg_sum(m[hg.edge], hg.node, n_nodes) * hg.dv_inv.unsqueeze(-1)


class HAttention(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.a_node = nn.Parameter(torch.empty(d))
        self.a_edge = nn.Parameter(torch.empty(d))
        nn.init.xavier_uniform_(self.a_node.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_edge.unsqueeze(0))

    def forward(self, x, hg: Hypergraph):
        # hyperedge representation = mean of its members (De^-1 H^T X)
        m = seg_sum(x[hg.node], hg.edge, hg.n_edges) * hg.de_inv.unsqueeze(-1)
        s = (x[hg.node] * self.a_node).sum(-1) + (m[hg.edge] * self.a_edge).sum(-1)
        alpha = seg_softmax(F.leaky_relu(s, 0.2), hg.node, x.shape[0])
        return seg_sum(alpha.unsqueeze(-1) * m[hg.edge], hg.node, x.shape[0])


# --------------------------------------------------------------------------- #
class GAttention(nn.Module):

    def __init__(self, d: int, chunk: int = 1024, negative_slope: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.a_s = nn.Parameter(torch.empty(d))
        self.a_t = nn.Parameter(torch.empty(d))
        nn.init.xavier_uniform_(self.a_s.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_t.unsqueeze(0))
        self.chunk = chunk
        self.slope = negative_slope
        self.gamma = 0.1

    def forward(self, x):
        h = self.proj(x)                                   # (N, d)
        src = (h * self.a_s).sum(-1)                       # (N,)
        dst = (h * self.a_t).sum(-1)                       # (N,)
        outs = []
        for s in range(0, x.shape[0], self.chunk):
            e = src[s:s + self.chunk, None] + dst[None, :]  # (c, N)
            alpha = torch.softmax(F.leaky_relu(e, self.slope), dim=1)
            outs.append(alpha @ h)                          # (c, d)
        return x + self.gamma * torch.cat(outs, 0)


# --------------------------------------------------------------------------- #
class GraphConv(nn.Module):

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.lin(x))


class LAttention(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.a_s = nn.Parameter(torch.empty(d))
        self.a_t = nn.Parameter(torch.empty(d))
        nn.init.xavier_uniform_(self.a_s.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_t.unsqueeze(0))

    def forward(self, x, edge_index):
        src, dst = edge_index
        s = (x[src] * self.a_s).sum(-1) + (x[dst] * self.a_t).sum(-1)
        alpha = seg_softmax(F.leaky_relu(s, 0.2), dst, x.shape[0])
        return seg_sum(alpha.unsqueeze(-1) * x[src], dst, x.shape[0])


# --------------------------------------------------------------------------- #
class HiRS(nn.Module):

    def __init__(self, d_in: int, hidden: int, layers: int, k: int):
        super().__init__()
        self.layers = layers
        self.k = k
        dims = [hidden] * (layers + 1)      # the input is projected to `hidden` first
        self.conv = nn.ModuleList([HyperConv(dims[i], dims[i + 1]) for i in range(layers)])
        self.att = nn.ModuleList([HAttention(dims[i + 1]) for i in range(layers)])
        # two heads with independent weights: universal / private
        self.v_u, self.g_u, self.e_u = (VertexConv(hidden, hidden), GAttention(hidden),
                                        EdgeConv(hidden, hidden))
        self.v_p, self.g_p, self.e_p = (VertexConv(hidden, hidden), GAttention(hidden),
                                        EdgeConv(hidden, hidden))
        self.inp = nn.Linear(d_in, hidden)

    def forward(self, x, hgs=None):

        h = self.inp(x)
        for l in range(self.layers):
            hg = (build_hypergraph(h.detach().cpu().numpy(), self.k, h.device)
                  if hgs is None else hgs[l])
            h = F.relu(self.att[l](self.conv[l](h, hg), hg)) + h             
        n = h.shape[0]      # `hg` now holds H^L, i.e. the hypergraph of the last layer
        r_u = self.e_u(self.g_u(self.v_u(h, hg)), hg, n)
        r_p = self.e_p(self.g_p(self.v_p(h, hg)), hg, n)
        return F.relu(r_u), F.relu(r_p)


class HeRS(nn.Module):

    def __init__(self, d_in: int, hidden: int, layers: int):
        super().__init__()
        dims = [hidden] * (layers + 1)      # the input is projected to `hidden` first
        self.conv = nn.ModuleList([GraphConv(dims[i], dims[i + 1]) for i in range(layers)])
        self.att = nn.ModuleList([LAttention(dims[i + 1]) for i in range(layers)])
        self.g_u = nn.Sequential(GraphConv(hidden, hidden), GAttention(hidden))
        self.g_p = nn.Sequential(GraphConv(hidden, hidden), GAttention(hidden))
        self.inp = nn.Linear(d_in, hidden)

    def forward(self, x, adj, edge_index):
        h = self.inp(x)
        for l in range(len(self.conv)):
            h = F.relu(self.att[l](self.conv[l](h, adj), edge_index)) + h    
        r_u = self.g_u[1](self.g_u[0](h, adj))                              
        r_p = self.g_p[1](self.g_p[0](h, adj))                              
        return F.relu(r_u), F.relu(r_p)


# --------------------------------------------------------------------------- #
class PAlign(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.lm1 = nn.Linear(d, d)
        self.lm2 = nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, y, z):
        q, k, v = self.q(y), self.k(z), self.v(z)
        attn = torch.softmax(q @ k.t() * self.scale, dim=-1)
        f_mid = torch.sigmoid(attn @ v + self.lm1(z))          
        return f_mid * self.lm2(z)                              


class SMF(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.palign = PAlign(d)
        self.u_fuse = nn.Linear(2 * d, d)       
        self.p_fuse = nn.Linear(2 * d, d)       
        self.lm_u1 = nn.Linear(d, d)            
        self.lm_u2 = nn.Linear(2 * d, d)        
        self.lm_p1 = nn.Linear(d, d)            
        self.lm_p2 = nn.Linear(2 * d, d)        
        self.beta = nn.Linear(2 * d, d)         

    def forward(self, ru_h, rp_h, ru_g, rp_g):
        f_u = F.relu(self.u_fuse(torch.cat([ru_h + ru_g, ru_h * ru_g], -1)))
        f_p = F.relu(self.p_fuse(torch.cat([self.palign(rp_h, rp_g),
                                            self.palign(rp_g, rp_h)], -1)))

        h_u = self.lm_u2(torch.cat([self.lm_u1(f_u), f_u], -1))
        h_p = self.lm_p2(torch.cat([self.lm_p1(f_p), f_p], -1))
        beta = torch.sigmoid(self.beta(torch.cat([h_u, h_p], -1)))
        return beta * h_u + (1 - beta) * h_p, beta


# --------------------------------------------------------------------------- #
class DecGR(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, layers: int = 4, k: int = 5,
                 dropout: float = 0.3, lam: float = 0.3):
        super().__init__()
        self.layers, self.k, self.lam = layers, k, lam
        self.hirs = HiRS(d_in, hidden, layers, k)
        self.hers = HeRS(d_in, hidden, layers)
        self.smf = SMF(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))          

        nn.init.normal_(self.cls[-1].weight, std=0.01)
        nn.init.zeros_(self.cls[-1].bias)

    def forward(self, x, adj, edge_index, hgs):
        ru_h, rp_h = self.hirs(self.drop(x), hgs)
        ru_g, rp_g = self.hers(self.drop(x), adj, edge_index)
        h_final, beta = self.smf(ru_h, rp_h, ru_g, rp_g)
        logits = self.cls(self.drop(h_final))
        return logits, {"ru_h": ru_h, "rp_h": rp_h, "ru_g": ru_g, "rp_g": rp_g,
                        "beta": beta, "h": h_final}

    def decoupling_loss(self, out):
        ru_h, rp_h, ru_g, rp_g = out["ru_h"], out["rp_h"], out["ru_g"], out["rp_g"]
        intra = (1 - cos_mean(ru_h, ru_g).abs())
        inter = (cos_mean(ru_h, rp_h) + cos_mean(ru_h, rp_g)
                 + cos_mean(ru_g, rp_h) + cos_mean(ru_g, rp_g))
        return intra + self.lam * inter

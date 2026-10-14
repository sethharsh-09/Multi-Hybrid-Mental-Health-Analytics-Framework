"""
MindGraphNet - Graph Learning & Graph Attention Network (GAT) Module
Constructs heterogeneous user-post-topic interaction graphs and learns topology-aware node representations.
"""
from typing import Tuple, Dict, List, Optional
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import REDUCED_EMB_DIM, GRAPH_EMB_DIM

class GraphAttentionLayer(nn.Module):
    """
    Native PyTorch multi-head Graph Attention Layer (Velickovic et al., ICLR 2018).
    Pure PyTorch implementation guaranteeing 100% portable execution on all platforms.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.empty(size=(n_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [N, in_features]
        # adj: [N, N] adjacency matrix (dense or binary)
        N = h.size(0)
        
        # Project: [n_heads, N, out_features]
        Wh = torch.matmul(h.unsqueeze(0), self.W)

        # Self-attention score computation
        Wh1 = torch.matmul(Wh, self.a[:, :self.out_features, :]) # [n_heads, N, 1]
        Wh2 = torch.matmul(Wh, self.a[:, self.out_features:, :]) # [n_heads, N, 1]
        e = Wh1 + Wh2.transpose(1, 2) # [n_heads, N, N]
        e = self.leaky_relu(e)

        # Mask non-connected edges (-infinity)
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).expand(self.n_heads, -1, -1)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        # Message passing aggregation
        h_prime = torch.matmul(attention, Wh) # [n_heads, N, out_features]

        if self.concat:
            return h_prime.permute(1, 0, 2).contiguous().view(N, self.n_heads * self.out_features)
        else:
            return h_prime.mean(dim=0)


class MentalHealthGAT(nn.Module):
    def __init__(self, in_dim: int = REDUCED_EMB_DIM, hidden_dim: int = 48, out_dim: int = GRAPH_EMB_DIM, heads: int = 4):
        super().__init__()
        self.gat1 = GraphAttentionLayer(in_dim, hidden_dim // heads, n_heads=heads, dropout=0.2, concat=True)
        self.gat2 = GraphAttentionLayer(hidden_dim, out_dim, n_heads=1, dropout=0.2, concat=False)
        self.decoder = nn.Linear(out_dim, in_dim) # Self-supervised reconstruction head

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = F.elu(self.gat1(x, adj))
        node_embs = self.gat2(h1, adj)
        recon_x = self.decoder(node_embs)
        return node_embs, recon_x


class GraphLearningEngine:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = nx.MultiDiGraph()
        self.node_index: Dict[str, int] = {}
        self.post_ids: List[str] = []
        self.gat_model: Optional[MentalHealthGAT] = None

    def build_graph(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """
        Builds heterogeneous multi-relational interaction graph.
        """
        self.G = nx.MultiDiGraph()
        
        # 1. Post Nodes
        for _, row in df.iterrows():
            self.G.add_node(row["post_id"], ntype="post", topic=row["topic"])

        # 2. Author Nodes & AUTHORS edges
        for _, row in df.iterrows():
            author = row["author_hash"]
            self.G.add_node(author, ntype="user")
            self.G.add_edge(author, row["post_id"], etype="AUTHORS")

        # 3. REPLIES_TO edges
        for _, row in df.iterrows():
            if pd.notna(row["replies_to"]) and row["replies_to"] in self.G:
                self.G.add_edge(row["post_id"], row["replies_to"], etype="REPLIES_TO")

        # 4. SAME_TOPIC edges (connect posts sharing same topic for community diffusion)
        for topic, grp in df.groupby("topic"):
            pids = grp["post_id"].tolist()
            for i in range(len(pids) - 1):
                self.G.add_edge(pids[i], pids[i+1], etype="SAME_TOPIC")

        return self.G

    def train_graph_embeddings(self, df: pd.DataFrame, initial_post_feats: np.ndarray, 
                               epochs: int = 35) -> np.ndarray:
        """
        Learns structural node representations via self-supervised graph attention training.
        """
        self.build_graph(df)
        self.post_ids = df["post_id"].tolist()
        author_ids = sorted(list({n for n, d in self.G.nodes(data=True) if d.get("ntype") == "user"}))
        all_nodes = self.post_ids + author_ids
        self.node_index = {nid: i for i, nid in enumerate(all_nodes)}
        
        N = len(all_nodes)
        feat_dim = initial_post_feats.shape[1]
        
        # Build initial feature matrix
        X = np.zeros((N, feat_dim), dtype=np.float32)
        for i, pid in enumerate(self.post_ids):
            X[self.node_index[pid]] = initial_post_feats[i]
            
        # Author nodes initialized with mean of their written posts
        for aid in author_ids:
            author_posts = (df["author_hash"] == aid).values
            if author_posts.any():
                X[self.node_index[aid]] = initial_post_feats[author_posts].mean(axis=0)

        # Adjacency matrix (symmetrized with self-loops)
        adj = np.eye(N, dtype=np.float32)
        for u, v in self.G.edges():
            if u in self.node_index and v in self.node_index:
                iu, iv = self.node_index[u], self.node_index[v]
                adj[iu, iv] = 1.0
                adj[iv, iu] = 1.0

        x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        adj_t = torch.tensor(adj, dtype=torch.float32).to(self.device)

        self.gat_model = MentalHealthGAT(in_dim=feat_dim, out_dim=GRAPH_EMB_DIM).to(self.device)
        optimizer = torch.optim.Adam(self.gat_model.parameters(), lr=0.01, weight_decay=1e-4)

        self.gat_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            node_embs, recon_x = self.gat_model(x_t, adj_t)
            # Self-supervised feature reconstruction + smoothness loss
            loss_recon = F.mse_loss(recon_x, x_t)
            loss_smooth = torch.mean(torch.sum((node_embs.unsqueeze(1) - node_embs.unsqueeze(0))**2 * adj_t.unsqueeze(-1), dim=(0, 1))) * 0.001
            total_loss = loss_recon + loss_smooth
            total_loss.backward()
            optimizer.step()

        self.gat_model.eval()
        with torch.no_grad():
            node_embs, _ = self.gat_model(x_t, adj_t)
            all_embs = node_embs.cpu().numpy()

        post_graph_embs = np.vstack([all_embs[self.node_index[pid]] for pid in self.post_ids])
        return post_graph_embs

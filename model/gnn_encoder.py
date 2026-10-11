"""
gnn_encoder.py — Graph Neural Network topology encoder.

Architecture:
  2-layer GraphSAGE encoder over the server cluster graph.
  Node features: (cpu_util, mem_util, session_ratio, load, drop_prob, alive, failure_rate)
  Edge features: (latency, bandwidth, utilization, packet_loss)

  Output:
    - per_node_embed: (N, hidden_dim) — used to score migration targets
    - global_embed:  (hidden_dim,)   — used as part of RL state

Design choice (from paper): 2-hop neighborhood = O(degree²) complexity,
independent of cluster size N. Policy remains scalable as cluster grows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    GraphSAGE-based encoder.

    Input:
        x:          (N, node_feat_dim) node features
        edge_index: (2, E) COO adjacency
        edge_attr:  (E, edge_feat_dim) edge features (projected to node space)

    Output:
        node_embed: (N, hidden_dim)
        graph_embed: (hidden_dim,)  — mean pool
    """

    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Project edge features to node feature space (added as node context)
        self.edge_proj = nn.Linear(edge_feat_dim, node_feat_dim)

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,           # (N, node_feat_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_feat_dim)
        batch: torch.Tensor = None,  # (N,) for batched graphs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Aggregate edge features into source nodes
        # Simple approach: scatter edge features to src nodes
        if edge_attr is not None and edge_index.shape[1] > 0:
            src_nodes = edge_index[0]  # (E,)
            edge_context = self.edge_proj(edge_attr)  # (E, node_feat_dim)
            # Mean-aggregate edge context per node
            node_edge_ctx = torch.zeros(x.shape[0], x.shape[1],
                                        device=x.device, dtype=x.dtype)
            node_edge_ctx.scatter_add_(0,
                src_nodes.unsqueeze(1).expand_as(edge_context),
                edge_context)
            x = x + node_edge_ctx

        # Project to hidden dim
        h = F.relu(self.input_proj(x))

        # Graph convolutions with residual connections
        for conv in self.convs:
            h_new = F.relu(conv(h, edge_index))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new   # residual

        h = self.norm(h)
        node_embed = self.out_proj(h)   # (N, hidden_dim)

        # Global graph embedding
        if batch is not None:
            graph_embed = global_mean_pool(node_embed, batch)   # (B, hidden_dim)
        else:
            graph_embed = node_embed.mean(dim=0)  # (hidden_dim,)

        return node_embed, graph_embed


class MigrationScorer(nn.Module):
    """
    Given a source node embedding and candidate target embeddings,
    scores each candidate target for migration suitability.

    Input:
        src_embed:      (hidden_dim,) — source server embedding
        target_embeds:  (K, hidden_dim) — K candidate neighbors
        session_feat:   (session_feat_dim,) — session features
        graph_embed:    (hidden_dim,) — global cluster state

    Output:
        scores: (K+1,) — score for [stay, migrate_to_v1, ..., migrate_to_vK]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        session_feat_dim: int = 6,
    ):
        super().__init__()
        input_dim = hidden_dim * 3 + session_feat_dim  # src + target + global + session

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.stay_head = nn.Linear(hidden_dim * 2 + session_feat_dim, 1)

    def forward(
        self,
        src_embed: torch.Tensor,        # (hidden_dim,)
        target_embeds: torch.Tensor,    # (K, hidden_dim)
        session_feat: torch.Tensor,     # (session_feat_dim,)
        graph_embed: torch.Tensor,      # (hidden_dim,)
    ) -> torch.Tensor:
        K = target_embeds.shape[0]

        # "Stay" score
        stay_input = torch.cat([src_embed, graph_embed, session_feat])
        stay_score = self.stay_head(stay_input)  # (1,)

        # Migration scores
        src_exp = src_embed.unsqueeze(0).expand(K, -1)    # (K, hidden_dim)
        graph_exp = graph_embed.unsqueeze(0).expand(K, -1)
        sess_exp = session_feat.unsqueeze(0).expand(K, -1)

        migrate_input = torch.cat([src_exp, target_embeds, graph_exp, sess_exp], dim=1)
        migrate_scores = self.mlp(migrate_input).squeeze(-1)  # (K,)

        return torch.cat([stay_score.squeeze(), migrate_scores])  # (K+1,)

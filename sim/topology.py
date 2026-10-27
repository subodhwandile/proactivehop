"""
topology.py — Server cluster topology graph generator.

Models a real-time server cluster as a weighted undirected graph:
  - Nodes: server instances (capacity, current load, sessions)
  - Edges: network links (latency, bandwidth, loss rate)

Supports fat-tree, random geometric, and scale-free topologies
to test policy generalization across diverse cluster configurations.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerNode:
    node_id: int
    cpu_capacity: float = 1.0          # normalized [0,1]
    mem_capacity: float = 1.0
    cpu_util: float = 0.0              # current utilization
    mem_util: float = 0.0
    active_sessions: int = 0
    max_sessions: int = 1000
    failure_rate: float = 0.001        # base Weibull scale (used to compute hazard_rate)
    hazard_rate: float = 0.0           # current per-step failure probability (load-coupled, updated each step)
    is_alive: bool = True
    region: int = 0                    # rack/zone label

    @property
    def load(self) -> float:
        return max(self.cpu_util, self.mem_util,
                   self.active_sessions / self.max_sessions)

    @property
    def drop_probability(self) -> float:
        """Sigmoid of excess load — probability a session drops this timestep.
        Calibrated for cleaner learning signal:
          load < 0.60 → near-zero drop prob
          load > 0.85 → high drop prob (~0.8)
        """
        if not self.is_alive:
            return 1.0
        excess = self.load - 0.72      # threshold centered at 72% load
        return float(1.0 / (1.0 + np.exp(-15.0 * excess)))

    def feature_vector(self) -> np.ndarray:
        """7-dim node feature vector for GNN input."""
        return np.array([
            self.cpu_util,
            self.mem_util,
            self.active_sessions / self.max_sessions,
            self.load,
            self.drop_probability,
            float(self.is_alive),
            self.hazard_rate,          # dynamic load-coupled failure hazard (replaces static failure_rate)
        ], dtype=np.float32)


@dataclass
class NetworkLink:
    src: int
    dst: int
    latency_ms: float = 1.0           # base latency
    bandwidth_gbps: float = 10.0
    utilization: float = 0.0          # [0,1]
    packet_loss: float = 0.0          # [0,1]

    @property
    def effective_latency(self) -> float:
        """Latency increases with congestion (M/D/1 queue model)."""
        rho = min(self.utilization, 0.99)
        return self.latency_ms / (1.0 - rho)

    def feature_vector(self) -> np.ndarray:
        """4-dim edge feature vector for GNN input."""
        return np.array([
            self.effective_latency / 100.0,  # normalize
            self.bandwidth_gbps / 100.0,
            self.utilization,
            self.packet_loss,
        ], dtype=np.float32)


class ClusterTopology:
    """
    Manages the server cluster graph.

    Graph semantics:
      G.nodes[i] = ServerNode
      G.edges[i,j] = NetworkLink
    """

    def __init__(
        self,
        topology: str = "fat_tree",
        n_servers: int = 16,
        seed: int = 42,
    ):
        self.n_servers = n_servers
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.graph = nx.Graph()
        self._build(topology)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, topology: str):
        if topology == "fat_tree":
            self._build_fat_tree()
        elif topology == "random_geo":
            self._build_random_geo()
        elif topology == "scale_free":
            self._build_scale_free()
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def _add_server(self, node_id: int, region: int = 0):
        max_sessions = int(self.rng.integers(500, 2000))
        node = ServerNode(
            node_id=node_id,
            cpu_capacity=1.0,
            mem_capacity=1.0,
            max_sessions=max_sessions,
            failure_rate=float(self.rng.uniform(0.0005, 0.002)),
            region=region,
        )
        self.graph.add_node(node_id, data=node)

    def _add_link(self, u: int, v: int, latency_ms: float = None, bw: float = None):
        if latency_ms is None:
            latency_ms = float(self.rng.uniform(0.5, 5.0))
        if bw is None:
            bw = float(self.rng.choice([1.0, 10.0, 25.0, 100.0]))
        link = NetworkLink(src=u, dst=v, latency_ms=latency_ms, bandwidth_gbps=bw)
        self.graph.add_edge(u, v, data=link)

    def _build_fat_tree(self):
        """
        Simplified fat-tree: k=4 → 16 servers, 8 aggregation, 4 core switches.
        Only server nodes are modeled; switch hops contribute to link latency.
        """
        k = max(4, int(np.sqrt(self.n_servers)))
        n = k * k  # total server nodes
        self.n_servers = n

        for i in range(n):
            region = i // k
            self._add_server(i, region=region)

        # Intra-rack links (low latency, high BW)
        for rack in range(k):
            servers = list(range(rack * k, (rack + 1) * k))
            for i in range(len(servers)):
                for j in range(i + 1, len(servers)):
                    self._add_link(servers[i], servers[j],
                                   latency_ms=float(self.rng.uniform(0.1, 0.5)),
                                   bw=25.0)

        # Inter-rack links (higher latency)
        for rack_a in range(k):
            for rack_b in range(rack_a + 1, k):
                # One cross-rack link per rack pair
                rep_a = rack_a * k
                rep_b = rack_b * k
                self._add_link(rep_a, rep_b,
                               latency_ms=float(self.rng.uniform(1.0, 5.0)),
                               bw=10.0)

    def _build_random_geo(self):
        """Random geometric graph — simulates geographically distributed servers."""
        G = nx.random_geometric_graph(self.n_servers, radius=0.4,
                                      seed=int(self.seed))
        for i in G.nodes():
            self._add_server(i, region=i % 4)
        for u, v in G.edges():
            pos_u = np.array(G.nodes[u]['pos'])
            pos_v = np.array(G.nodes[v]['pos'])
            dist = float(np.linalg.norm(pos_u - pos_v))
            self._add_link(u, v, latency_ms=dist * 20.0)

    def _build_scale_free(self):
        """Barabási–Albert scale-free graph — simulates hub-and-spoke clusters."""
        G = nx.barabasi_albert_graph(self.n_servers, m=3, seed=int(self.seed))
        for i in G.nodes():
            self._add_server(i, region=G.degree(i) // 5)
        for u, v in G.edges():
            self._add_link(u, v)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: int) -> ServerNode:
        return self.graph.nodes[node_id]['data']

    def get_link(self, u: int, v: int) -> Optional[NetworkLink]:
        if self.graph.has_edge(u, v):
            return self.graph.edges[u, v]['data']
        return None

    def neighbors(self, node_id: int) -> list[int]:
        return list(self.graph.neighbors(node_id))

    def node_features(self) -> np.ndarray:
        """Returns (N, 7) node feature matrix."""
        N = self.graph.number_of_nodes()
        F = np.zeros((N, 7), dtype=np.float32)
        for i in range(N):
            F[i] = self.get_node(i).feature_vector()
        return F

    def edge_index(self) -> np.ndarray:
        """Returns (2, E) COO edge index for PyTorch Geometric."""
        edges = list(self.graph.edges())
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        return np.array([src, dst], dtype=np.int64)

    def edge_features(self) -> np.ndarray:
        """Returns (2E, 4) edge feature matrix (undirected → doubled)."""
        edges = list(self.graph.edges())
        if not edges:
            return np.zeros((0, 4), dtype=np.float32)
        feats = []
        for u, v in edges:
            f = self.graph.edges[u, v]['data'].feature_vector()
            feats.append(f)
            feats.append(f)  # both directions
        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    # State mutation (called by simulator each timestep)
    # ------------------------------------------------------------------

    def update_loads(self, load_delta: dict[int, float], session_delta: dict[int, int]):
        """Apply load changes from the simulator."""
        for node_id, delta in load_delta.items():
            node = self.get_node(node_id)
            node.cpu_util = float(np.clip(node.cpu_util + delta, 0.0, 1.0))
            node.mem_util = float(np.clip(node.mem_util + delta * 0.8, 0.0, 1.0))
        for node_id, delta in session_delta.items():
            node = self.get_node(node_id)
            node.active_sessions = max(0, node.active_sessions + delta)

    def apply_failure(self, node_id: int):
        node = self.get_node(node_id)
        node.is_alive = False
        node.cpu_util = 1.0
        node.mem_util = 1.0

    def recover_node(self, node_id: int):
        node = self.get_node(node_id)
        node.is_alive = True
        node.cpu_util = 0.0
        node.mem_util = 0.0
        node.active_sessions = 0

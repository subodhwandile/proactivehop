"""
baselines.py — All comparison baselines for the paper evaluation.

Baselines:
  1. NoMigration     — sessions never proactively moved; reactive reconnect only
  2. ReactiveOnly    — same as NoMigration (explicit label for paper clarity)
  3. ThresholdPolicy — migrate when server load > fixed threshold (heuristic)
  4. RandomPolicy    — randomly trigger migration with fixed probability
  5. OraclePolicy    — knows exactly which server will fail next (upper bound)

Each baseline implements the same interface as PPOAgent.select_action:
    select_action(obs: dict, sim: ClusterSimulator) -> dict[session_id -> target]

Returns migration decisions dict directly (not binary action array).
"""

import numpy as np
from sim.cluster import ClusterSimulator
from sim.session import SessionStatus


class NoMigrationPolicy:
    """Baseline 1 & 2: Never proactively migrate. Pure reactive reconnect."""
    name = "No Migration (Reactive)"

    def select_action(self, obs: dict, sim: ClusterSimulator) -> dict:
        return {}  # no decisions → all drops handled reactively


class ThresholdPolicy:
    """
    Baseline 3: Trigger pre-migration when server load exceeds threshold.
    Classic load-balancer heuristic — no ML.
    """
    name = "Load Threshold (Heuristic)"

    def __init__(self, load_threshold: float = 0.80):
        self.threshold = load_threshold

    def select_action(self, obs: dict, sim: ClusterSimulator) -> dict:
        decisions = {}
        for server_id in range(sim.n):
            node = sim.topo.get_node(server_id)
            if not node.is_alive or node.load <= self.threshold:
                continue

            # Find lowest-load neighbor
            neighbors = sim.topo.neighbors(server_id)
            live = [nb for nb in neighbors
                    if sim.topo.get_node(nb).is_alive
                    and sim.topo.get_node(nb).load < 0.6]
            if not live:
                continue
            target = min(live, key=lambda nb: sim.topo.get_node(nb).load)

            sessions = [
                sim.sessions[sid]
                for sid in sim.server_sessions[server_id]
                if sid in sim.sessions
                and sim.sessions[sid].status == SessionStatus.ACTIVE
            ]
            sessions.sort(key=lambda s: s.state.state_payload_kb, reverse=True)
            for s in sessions[:20]:
                decisions[s.session_id] = target

        return decisions


class RandomPolicy:
    """
    Baseline 4: Randomly trigger migration with fixed probability per server.
    Models the "random migration as noise floor" baseline.
    """
    name = "Random Policy"

    def __init__(self, trigger_prob: float = 0.1, seed: int = 0):
        self.prob = trigger_prob
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: dict, sim: ClusterSimulator) -> dict:
        decisions = {}
        for server_id in range(sim.n):
            if self.rng.random() > self.prob:
                continue
            node = sim.topo.get_node(server_id)
            if not node.is_alive:
                continue

            neighbors = sim.topo.neighbors(server_id)
            live = [nb for nb in neighbors if sim.topo.get_node(nb).is_alive]
            if not live:
                continue
            target = int(self.rng.choice(live))

            sessions = [
                sim.sessions[sid]
                for sid in sim.server_sessions[server_id]
                if sid in sim.sessions
                and sim.sessions[sid].status == SessionStatus.ACTIVE
            ]
            for s in sessions[:2]:
                decisions[s.session_id] = target

        return decisions



class NeighbourAwareThreshold:
    """
    Neighbour-aware threshold: migrate when ℓ_v ≥ θ_src AND mean(neighbour loads) < φ_dst.
    The second condition prevents sending sessions to already-loaded neighbours,
    making it a stronger per-server heuristic than plain ThresholdPolicy.
    """
    name = "Neighbour-Aware Threshold"

    def __init__(self, src_threshold: float = 0.70, dst_threshold: float = 0.60):
        self.src_threshold = src_threshold
        self.dst_threshold = dst_threshold

    def select_action(self, obs: dict, sim: ClusterSimulator) -> dict:
        decisions = {}
        for server_id in range(sim.n):
            node = sim.topo.get_node(server_id)
            if not node.is_alive or node.load < self.src_threshold:
                continue

            neighbors = sim.topo.neighbors(server_id)
            live = [nb for nb in neighbors
                    if sim.topo.get_node(nb).is_alive
                    and sim.topo.get_node(nb).load < self.dst_threshold]
            # Also require mean neighbour load below dst_threshold
            if not live:
                continue
            nb_loads = [sim.topo.get_node(nb).load for nb in neighbors
                        if sim.topo.get_node(nb).is_alive]
            if nb_loads and np.mean(nb_loads) >= self.dst_threshold:
                continue

            target = min(live, key=lambda nb: sim.topo.get_node(nb).load)
            sessions = [
                sim.sessions[sid]
                for sid in sim.server_sessions[server_id]
                if sid in sim.sessions
                and sim.sessions[sid].status == SessionStatus.ACTIVE
            ]
            for s in sessions[:20]:
                decisions[s.session_id] = target

        return decisions


class OraclePolicy:
    """
    Baseline 5: Oracle — knows exactly which server will fail before it fails.
    Provides theoretical upper bound on catch rate.

    In the simulator, server failure probability is computed each step.
    Oracle checks this and pre-migrates sessions when p_fail > epsilon.
    """
    name = "Oracle (Upper Bound)"

    def __init__(self, fail_prob_threshold: float = 0.005):
        self.threshold = fail_prob_threshold

    def select_action(self, obs: dict, sim: ClusterSimulator) -> dict:
        decisions = {}
        for server_id in range(sim.n):
            node = sim.topo.get_node(server_id)
            if not node.is_alive:
                continue

            # Oracle has access to drop_probability (normally unobservable)
            if node.drop_probability < self.threshold:
                continue

            # Pre-migrate ALL sessions on this server
            neighbors = sim.topo.neighbors(server_id)
            live = [nb for nb in neighbors
                    if sim.topo.get_node(nb).is_alive
                    and sim.topo.get_node(nb).load < 0.9]
            if not live:
                # Try any alive server
                live = [i for i in range(sim.n)
                        if i != server_id and sim.topo.get_node(i).is_alive]
            if not live:
                continue

            # Spread load: sort targets by load
            live.sort(key=lambda nb: sim.topo.get_node(nb).load)

            sessions = [
                sim.sessions[sid]
                for sid in sim.server_sessions[server_id]
                if sid in sim.sessions
                and sim.sessions[sid].status == SessionStatus.ACTIVE
            ]
            for idx, s in enumerate(sessions):
                target = live[idx % len(live)]
                decisions[s.session_id] = target

        return decisions

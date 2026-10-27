"""
cluster.py — Main simulation environment for the server cluster.

Drives:
  - Workload generation (Poisson session arrivals, heavy-tail durations)
  - Server load evolution (bursty traffic, correlated spikes)
  - Failure injection (Weibull-distributed MTBF)
  - Migration execution (speculative + reactive)
  - Metric collection
"""

import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from sim.topology import ClusterTopology
from sim.session import Session, SessionStatus
from sim.trace_loader import AlibabaTraceLoader


@dataclass
class SimConfig:
    # Cluster
    topology: str = "fat_tree"
    n_servers: int = 16
    seed: int = 42

    # Workload
    session_arrival_rate: float = 50.0    # sessions/second across cluster
    session_duration_mean: float = 60.0   # steps (simulation seconds) per session
    session_duration_alpha: float = 1.5   # Pareto shape

    # Load model
    base_load: float = 0.3                # baseline CPU utilization
    spike_probability: float = 0.05       # per-server per-step spike chance
    spike_magnitude: float = 0.4          # load increase during spike
    spike_duration: float = 10.0          # seconds

    # Failures
    server_mtbf: float = 3600.0           # mean time between failures (seconds)
    server_mttr: float = 60.0             # mean time to recover (seconds)
    enable_failures: bool = True

    # Migration
    speculative_push_latency_ms: float = 5.0   # state push time (network)
    reactive_reconnect_latency_ms_mean: float = 400.0
    reactive_reconnect_latency_ms_std: float = 150.0
    speculative_reconnect_latency_ms_mean: float = 5.0
    speculative_reconnect_latency_ms_std: float = 2.0

    # Simulation
    timestep_sec: float = 1.0             # simulation step = 1 second
    max_steps: int = 3600                 # 1 hour of simulation

    # Trace-driven workload (optional)
    use_trace: bool = False               # if True, use Alibaba 2018 trace arrivals


@dataclass
class StepMetrics:
    step: int = 0
    drops_reactive: int = 0       # drops recovered reactively
    drops_speculative: int = 0    # drops caught by speculative migration
    drops_unhandled: int = 0      # drops with no pre-migration
    false_positive_migrations: int = 0   # speculative push, no drop occurred
    true_positive_migrations: int = 0    # speculative push, drop DID occur
    avg_reconnect_latency_ms: float = 0.0
    total_migrations: int = 0
    migration_bandwidth_kb: float = 0.0
    active_sessions: int = 0
    overloaded_servers: int = 0
    failed_servers: int = 0


class ClusterSimulator:
    """
    Discrete-time simulator. Each step = timestep_sec seconds of simulated time.
    The RL agent (or a baseline policy) calls step() with migration decisions.
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.topo = ClusterTopology(
            topology=config.topology,
            n_servers=config.n_servers,
            seed=config.seed,
        )
        self.n = self.topo.graph.number_of_nodes()

        # Trace-driven arrivals (Alibaba 2018)
        self._trace: AlibabaTraceLoader | None = None
        if config.use_trace:
            self._trace = AlibabaTraceLoader(target_mean=config.session_arrival_rate)

        # Session registry
        self.sessions: dict[str, Session] = {}
        self.server_sessions: dict[int, set] = defaultdict(set)

        # Spike state per server
        self._spike_remaining: dict[int, float] = defaultdict(float)

        # Failure state
        self._recovery_remaining: dict[int, float] = defaultdict(float)

        # Metrics history
        self.history: list[StepMetrics] = []
        self.current_step = 0

        # Persistent migration state:
        # session_id -> old_server_id (populated per-step, cleared at start of each step)
        self._migrated_this_step: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset simulator and return initial observation."""
        self.__init__(self.cfg)
        self._seed_initial_sessions()
        return self._get_obs()

    def step(self, migration_decisions: dict[str, int]) -> tuple[dict, float, bool, dict]:
        """
        Advance simulation by one timestep.

        Args:
            migration_decisions: {session_id -> target_server_id}
                                  Policy output. Empty dict = no migrations.

        Returns:
            obs, reward, done, info
        """
        metrics = StepMetrics(step=self.current_step)
        self._migrated_this_step = {}  # reset per-step tracking

        # 1. Arrive new sessions
        self._arrive_sessions()

        # 2. Update server loads (spikes, decay, correlated noise)
        self._update_loads()

        # 3. Apply failure / recovery events
        if self.cfg.enable_failures:
            self._apply_failures(metrics)

        # 4. Execute persistent migrations (policy decisions)
        self._execute_persistent_migrations(migration_decisions, metrics)

        # 5. Evaluate which sessions drop this timestep
        self._evaluate_drops(metrics)

        # 6. Expire old sessions
        self._expire_sessions()

        # 7. Compute reward
        reward = self._compute_reward(metrics)

        # 8. Record metrics
        metrics.active_sessions = len(self.sessions)
        metrics.overloaded_servers = sum(
            1 for i in range(self.n)
            if self.topo.get_node(i).load > 0.8 and self.topo.get_node(i).is_alive
        )
        metrics.failed_servers = sum(
            1 for i in range(self.n)
            if not self.topo.get_node(i).is_alive
        )
        self.history.append(metrics)
        self.current_step += 1

        done = self.current_step >= self.cfg.max_steps
        obs = self._get_obs()
        return obs, reward, done, {"metrics": metrics}

    # ------------------------------------------------------------------
    # Workload generation
    # ------------------------------------------------------------------

    def _seed_initial_sessions(self):
        """Start with ~50 sessions per server."""
        n_initial = int(self.n * 50)
        for _ in range(n_initial):
            server_id = int(self.rng.integers(0, self.n))
            self._create_session(server_id)

    def _arrive_sessions(self):
        """Poisson arrivals (or trace-driven) distributed across live servers."""
        live_servers = [i for i in range(self.n) if self.topo.get_node(i).is_alive]
        if not live_servers:
            return
        if self._trace is not None:
            # Trace-driven: use real Alibaba arrival count for this step
            n_arrivals = int(round(self._trace.get_rate(self.current_step)))
        else:
            rate = self.cfg.session_arrival_rate * self.cfg.timestep_sec
            n_arrivals = int(self.rng.poisson(rate))
        for _ in range(n_arrivals):
            server_id = int(self.rng.choice(live_servers))
            node = self.topo.get_node(server_id)
            if node.active_sessions < node.max_sessions:
                self._create_session(server_id)

    def _create_session(self, server_id: int) -> Session:
        s = Session(home_server=server_id, current_server=server_id,
                    created_step=self.current_step, last_active_step=self.current_step)
        s.state.session_id = s.session_id
        s.state.user_id = s.user_id
        s.state.server_id = server_id
        self.sessions[s.session_id] = s
        self.server_sessions[server_id].add(s.session_id)
        self.topo.update_loads({}, {server_id: 1})
        return s

    def _expire_sessions(self):
        """Remove sessions that have exceeded their sampled lifetime (in simulation steps)."""
        expired = []
        for sid, s in self.sessions.items():
            # Use Pareto-distributed session lifetime in simulation steps
            duration = self.cfg.session_duration_mean * (
                self.rng.pareto(self.cfg.session_duration_alpha) + 1
            )
            s.last_active_step = self.current_step
            if s.age_seconds > duration:
                expired.append(sid)
        for sid in expired:
            s = self.sessions.pop(sid)
            self.server_sessions[s.current_server].discard(sid)
            self.topo.update_loads({}, {s.current_server: -1})

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    def _update_loads(self):
        load_delta = {}
        for i in range(self.n):
            node = self.topo.get_node(i)
            if not node.is_alive:
                continue

            # Decay toward base load
            decay = (self.cfg.base_load - node.cpu_util) * 0.05
            delta = decay

            # Correlated spike injection
            if self._spike_remaining[i] > 0:
                self._spike_remaining[i] -= self.cfg.timestep_sec
                delta += self.cfg.spike_magnitude * 0.1   # sustain spike
            elif self.rng.random() < self.cfg.spike_probability:
                self._spike_remaining[i] = self.cfg.spike_duration
                delta += self.cfg.spike_magnitude

            # Small Gaussian noise
            delta += float(self.rng.normal(0, 0.02))
            load_delta[i] = delta

        self.topo.update_loads(load_delta, {})

    # ------------------------------------------------------------------
    # Failure model
    # ------------------------------------------------------------------

    def _apply_failures(self, metrics: StepMetrics):
        dt = self.cfg.timestep_sec
        mtbf = self.cfg.server_mtbf
        for i in range(self.n):
            node = self.topo.get_node(i)
            if not node.is_alive:
                self._recovery_remaining[i] -= dt
                if self._recovery_remaining[i] <= 0:
                    self.topo.recover_node(i)
            else:
                # Load-coupled Weibull hazard: overloaded servers fail more often
                p_fail_base = 1.0 - np.exp(-(dt / mtbf) ** 1.5)
                p_fail = p_fail_base + 0.005 * max(0.0, node.load - 0.70)
                p_fail = min(p_fail, 0.05)   # cap at 5% per step
                node.hazard_rate = p_fail * 20.0  # scale to [0,1] range for GNN feature
                if self.rng.random() < p_fail:
                    self.topo.apply_failure(i)
                    self._recovery_remaining[i] = float(
                        self.rng.exponential(self.cfg.server_mttr)
                    )

    # ------------------------------------------------------------------
    # Migration execution
    # ------------------------------------------------------------------

    def _execute_persistent_migrations(
        self,
        decisions: dict[str, int],
        metrics: StepMetrics,
    ):
        """
        Persistent migration: immediately move each session to its target.
        Track old server in _migrated_this_step for counterfactual TP/FP eval.
        """
        for session_id, target in decisions.items():
            if session_id not in self.sessions:
                continue
            s = self.sessions[session_id]
            if s.status != SessionStatus.ACTIVE:
                continue
            src = s.current_server
            if not self.topo.get_node(src).is_alive:
                continue
            if not self.topo.get_node(target).is_alive:
                continue

            bw_cost = s.state.delta_size_kb(s.prev_checkpoint_hash)
            metrics.migration_bandwidth_kb += bw_cost
            metrics.total_migrations += 1

            self._move_session(session_id, src, target)
            self._migrated_this_step[session_id] = src

    # ------------------------------------------------------------------
    # Drop evaluation
    # ------------------------------------------------------------------

    def _evaluate_drops(self, metrics: StepMetrics):
        """
        Evaluate session drops for this timestep.

        For sessions moved this step (persistent migration):
          - Sample counterfactual p_drop at the OLD server
          - TP if old server would have dropped it; FP otherwise

        For sessions not moved:
          - Sample p_drop at current server normally
        """
        reconnect_latencies = []

        for sid, s in list(self.sessions.items()):
            if sid in self._migrated_this_step:
                # Counterfactual: would the old server have dropped this session?
                old_server = self._migrated_this_step[sid]
                old_node = self.topo.get_node(old_server)
                p_drop = old_node.drop_probability
                p_drop_session = min(1.0, max(0.0,
                    p_drop + float(self.rng.normal(0, 0.01))))

                if self.rng.random() < p_drop_session:
                    # TP: migration was justified
                    lat = float(self.rng.normal(
                        self.cfg.speculative_reconnect_latency_ms_mean,
                        self.cfg.speculative_reconnect_latency_ms_std,
                    ))
                    lat = max(1.0, lat)
                    s.record_reconnect(lat, was_speculative=True)
                    metrics.true_positive_migrations += 1
                    reconnect_latencies.append(lat)
                else:
                    # FP: migration was unnecessary
                    metrics.false_positive_migrations += 1
            else:
                # Standard path: evaluate drop at current server
                node = self.topo.get_node(s.current_server)
                p_drop = node.drop_probability
                p_drop_session = min(1.0, max(0.0,
                    p_drop + float(self.rng.normal(0, 0.01))))

                if self.rng.random() < p_drop_session:
                    s.record_drop()
                    metrics.drops_unhandled += 1
                    self._reactive_reconnect(sid, s, metrics, reconnect_latencies)

        if reconnect_latencies:
            metrics.avg_reconnect_latency_ms = float(np.mean(reconnect_latencies))
        metrics.drops_reactive = metrics.drops_unhandled

    def _reactive_reconnect(self, sid, s, metrics, latency_list):
        """Session drops with no pre-migration — full cold-start reconnect."""
        lat = float(self.rng.normal(
            self.cfg.reactive_reconnect_latency_ms_mean,
            self.cfg.reactive_reconnect_latency_ms_std,
        ))
        lat = max(50.0, lat)
        s.record_reconnect(lat, was_speculative=False)

        # Find best available server
        live = [i for i in range(self.n)
                if self.topo.get_node(i).is_alive and i != s.current_server]
        if live:
            target = min(live, key=lambda i: self.topo.get_node(i).load)
            self._move_session(sid, s.current_server, target)
        latency_list.append(lat)

    def _move_session(self, sid: str, src: int, dst: int):
        s = self.sessions[sid]
        self.server_sessions[src].discard(sid)
        self.server_sessions[dst].add(sid)
        self.topo.update_loads({}, {src: -1, dst: 1})
        s.complete_migration(dst)

    # ------------------------------------------------------------------
    # Observation and reward
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        return {
            "node_features": self.topo.node_features(),          # (N, 7)
            "edge_index": self.topo.edge_index(),                 # (2, E)
            "edge_features": self.topo.edge_features(),           # (2E, 4)
            "n_nodes": self.n,
            "active_sessions": len(self.sessions),
        }

    def _compute_reward(self, m: StepMetrics) -> float:
        """
        Differential latency reward.

        Conceptually: reward = latency_saved_this_step - bandwidth_cost
          latency_saved = (reactive_latency - speculative_latency) * n_caught
          baseline: if speculative_caught sessions had been reactive:
                    they'd cost ~reactive_mean ms each
          actual: they cost ~speculative_mean ms each

        This properly credits the agent for IMPROVEMENT over no-migration.
        Bandwidth penalty is small relative to latency savings.
        """
        reactive_mean_ms = self.cfg.reactive_reconnect_latency_ms_mean
        speculative_mean_ms = self.cfg.speculative_reconnect_latency_ms_mean
        C_bw = 0.00005   # cost per KB bandwidth (small relative to latency)

        # Latency saved by speculative catches
        latency_saved = (m.true_positive_migrations
                         * (reactive_mean_ms - speculative_mean_ms))

        # Small penalty for false-positive bandwidth (not huge)
        bw_penalty = C_bw * m.migration_bandwidth_kb

        reward = latency_saved - bw_penalty
        return float(reward)

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self.history:
            return {}
        all_lat = []
        for s in self.sessions.values():
            all_lat.extend(s.reconnect_latencies_ms)
        total_drops = sum(m.drops_unhandled + m.true_positive_migrations for m in self.history)
        total_spec = sum(m.true_positive_migrations for m in self.history)
        total_unhandled = sum(m.drops_unhandled for m in self.history)
        total_fp = sum(m.false_positive_migrations for m in self.history)
        total_bw = sum(m.migration_bandwidth_kb for m in self.history)

        return {
            "total_drops": total_drops,
            "speculative_caught": total_spec,
            "unhandled_drops": total_unhandled,
            "false_positives": total_fp,
            "catch_rate": total_spec / max(total_drops, 1),
            "false_positive_rate": total_fp / max(total_spec + total_fp, 1),
            "total_migration_bw_mb": total_bw / 1024,
            "p50_reconnect_ms": float(np.percentile(all_lat, 50)) if all_lat else 0,
            "p95_reconnect_ms": float(np.percentile(all_lat, 95)) if all_lat else 0,
            "p99_reconnect_ms": float(np.percentile(all_lat, 99)) if all_lat else 0,
        }

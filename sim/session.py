"""
session.py — Session state model with delta-serialization.

A Session represents one persistent real-time connection (WebSocket/QUIC).
Session state is designed for efficient delta-push during speculative migration:
  - Full state: entire session context
  - Delta state: only changes since last checkpoint

This models the speculative pre-push mechanism from the paper:
  At t=0ms: policy predicts migration needed
  At t=5ms: delta state pushed to destination
  At t=X ms: client drops / reconnects → instant resume
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SessionStatus(Enum):
    ACTIVE = "active"
    MIGRATING = "migrating"      # speculative push in progress
    DROPPED = "dropped"          # connection lost
    RESUMED = "resumed"          # reconnected after drop
    TERMINATED = "terminated"


@dataclass
class SessionState:
    """Represents the serializable state of one session."""
    session_id: str
    user_id: str
    server_id: int
    created_at: float
    last_activity: float
    message_count: int = 0
    bytes_sent: int = 0
    bytes_recv: int = 0
    # Application-level state (simulated as a fixed-size payload)
    state_payload_kb: float = 0.0     # KB of state to transfer on migration
    checkpoint_hash: str = ""

    def delta_size_kb(self, prev_checkpoint_hash: str) -> float:
        """
        Simulates delta compression: if state unchanged → 3% of full state.
        If heavily changed → up to 30% of full state.
        In production this would be the actual diff size.
        """
        if prev_checkpoint_hash == self.checkpoint_hash:
            return self.state_payload_kb * 0.03
        h1 = int(hashlib.md5(self.checkpoint_hash.encode(), usedforsecurity=False).hexdigest(), 16)
        h2 = int(hashlib.md5(prev_checkpoint_hash.encode(), usedforsecurity=False).hexdigest(), 16)
        change_ratio = abs(h1 - h2) % 100 / 100.0
        return self.state_payload_kb * (0.03 + 0.27 * change_ratio)

    def full_size_kb(self) -> float:
        return self.state_payload_kb


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: "s0000000")
    user_id: str = field(default_factory=lambda: "u0000000")
    home_server: int = 0
    current_server: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    # Simulation step tracking (not real time)
    created_step: int = 0
    last_active_step: int = 0

    # Migration tracking
    migration_count: int = 0
    last_migration_at: float = 0.0
    speculative_target: Optional[int] = None
    speculative_push_at: float = 0.0

    # Drop / reconnect tracking
    drop_count: int = 0
    drop_times: list = field(default_factory=list)
    reconnect_latencies_ms: list = field(default_factory=list)

    # State
    state: SessionState = field(default_factory=lambda: SessionState(
        session_id="", user_id="", server_id=0,
        created_at=time.time(), last_activity=time.time(),
        state_payload_kb=100.0,
        checkpoint_hash="ck0000000",
    ))
    prev_checkpoint_hash: str = ""

    def update_activity(self):
        self.last_activity = time.time()
        self.state.message_count += 1
        self.state.bytes_sent += int(np.random.exponential(1024))
        self.state.bytes_recv += int(np.random.exponential(512))
        # Occasionally update state (creates larger deltas)
        if np.random.random() < 0.1:
            self.prev_checkpoint_hash = self.state.checkpoint_hash
            self.state.checkpoint_hash = hashlib.md5(
                self.state.checkpoint_hash.encode(), usedforsecurity=False
            ).hexdigest()[:8]

    def start_speculative_push(self, target_server: int):
        """Policy decided to proactively push state to target."""
        self.speculative_target = target_server
        self.speculative_push_at = time.time()
        self.status = SessionStatus.MIGRATING

    def complete_migration(self, actual_target: int):
        """Finalize migration (speculative or reactive)."""
        self.current_server = actual_target
        self.speculative_target = None
        self.migration_count += 1
        self.last_migration_at = time.time()
        self.status = SessionStatus.ACTIVE

    def record_drop(self):
        self.drop_count += 1
        self.drop_times.append(time.time())
        self.status = SessionStatus.DROPPED

    def record_reconnect(self, latency_ms: float, was_speculative: bool):
        """
        Reconnect latency:
          - Speculative (state pre-pushed):  2-10ms  (token check + state resume)
          - Reactive (cold start):           200-800ms (re-auth + state fetch)
        """
        self.reconnect_latencies_ms.append(latency_ms)
        self.status = SessionStatus.RESUMED

    @property
    def age_seconds(self) -> float:
        """Returns simulation age in steps (each step = 1 simulated second)."""
        return float(self.last_active_step - self.created_step)

    @property
    def feature_vector(self) -> np.ndarray:
        """6-dim session feature vector for RL state."""
        age = min(self.age_seconds / 3600.0, 1.0)         # normalized age (max 1h)
        idle = min((time.time() - self.last_activity) / 60.0, 1.0)
        mig_rate = min(self.migration_count / 10.0, 1.0)
        drop_rate = min(self.drop_count / 5.0, 1.0)
        state_size = min(self.state.state_payload_kb / 1024.0, 1.0)
        delta_ratio = self.state.delta_size_kb(self.prev_checkpoint_hash) / max(self.state.full_size_kb(), 1.0)
        return np.array([age, idle, mig_rate, drop_rate, state_size, delta_ratio],
                        dtype=np.float32)

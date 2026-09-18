from __future__ import annotations

from threading import Event
import time
from typing import Any

from .contracts import Phase
from .get5_control import Get5ControlState
from .runtime import QueueAborted
from .selfplay import MatchCompletionV1, RolloutEnvelopeV1, SelfPlayController


class SelfPlayWorker:
    def __init__(
        self,
        *,
        transport: Any,
        runtime: Any,
        controller: SelfPlayController,
        control_state: Get5ControlState | None = None,
        rollout_horizon: int = 1024,
        wait_timeout_ms: int = 100,
    ) -> None:
        if not isinstance(rollout_horizon, int) or isinstance(rollout_horizon, bool) or rollout_horizon < 1:
            raise ValueError("rollout_horizon must be a positive integer")
        if not isinstance(wait_timeout_ms, int) or isinstance(wait_timeout_ms, bool) or wait_timeout_ms < 0:
            raise ValueError("wait_timeout_ms must be a non-negative integer")
        self.transport = transport
        self.runtime = runtime
        self.controller = controller
        self.control_state = control_state or Get5ControlState()
        self.rollout_horizon = rollout_horizon
        self.wait_timeout_ms = wait_timeout_ms

    def run(self, stop_event: Event, rollout_queue: Any, metrics_queue: Any) -> None:
        pending_controller_boundaries: tuple[Any, ...] = ()
        try:
            while not stop_event.is_set():
                try:
                    new_boundaries = self.control_state.consume_available(self.transport)
                    boundaries = pending_controller_boundaries + new_boundaries
                    pending_controller_boundaries = ()
                    self._activate_staged_policy(new_boundaries)
                    self._apply_runtime_boundaries(new_boundaries)
                    if not self.transport.wait_for_observation(self.wait_timeout_ms):
                        pending_controller_boundaries = boundaries
                        if self._transport_aborted():
                            break
                        continue
                    read_started_ns = time.perf_counter_ns()
                    observation = self.transport.try_read_observation()
                    if observation is not None:
                        # Inference is deliberately latest-state based.  If the
                        # server produced faster than the model, applying every
                        # queued packet would publish actions for snapshots that
                        # have already fallen out of the bridge's identity ring.
                        observation = self._read_latest_observation(observation)
                    read_completed_ns = time.perf_counter_ns()
                    late_boundaries = self.control_state.consume_available(self.transport)
                    if late_boundaries:
                        self._activate_staged_policy(late_boundaries)
                        self._apply_runtime_boundaries(late_boundaries)
                        boundaries += late_boundaries
                except QueueAborted:
                    break
                if observation is None:
                    pending_controller_boundaries = boundaries
                    continue
                if self._observation_precedes_round_start(observation, late_boundaries):
                    # Get5 can report `live` before the engine emits the first
                    # real round_start.  The bridge resets its snapshot ring at
                    # that event, so a pre-boundary observation can never accept
                    # an action packet.  Keep the boundaries for the next valid
                    # observation and do not train on this stale packet.
                    pending_controller_boundaries = boundaries
                    continue
                runtime_started_ns = read_completed_ns
                try:
                    action = self.runtime.process_observation_batch(observation)
                except QueueAborted:
                    break
                runtime_finished_ns = time.perf_counter_ns()
                timing = self._runtime_timing()
                inference_started_ns = int(timing.get("inference_started_ns", runtime_started_ns))
                inference_completed_ns = int(timing.get("inference_completed_ns", runtime_finished_ns))
                publish_started_ns = int(timing.get("publish_started_ns", inference_completed_ns))
                publish_completed_ns = int(timing.get("publish_completed_ns", runtime_finished_ns))
                try:
                    observed_phase = Phase(int(observation.observations[0][5]))
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError("self-play observation does not contain a valid Get5 phase") from error
                if (
                    self.controller.phase is Phase.LIVE
                    and observed_phase in {Phase.WARMUP, Phase.KNIFE}
                ):
                    self.controller.apply_boundaries(boundaries)
                    if self.controller.status == "finished":
                        stop_event.set()
                        break
                    metrics_queue.put(
                        self._metrics(
                            observation,
                            action,
                            read_started_ns=read_started_ns,
                            read_completed_ns=read_completed_ns,
                            inference_started_ns=inference_started_ns,
                            inference_completed_ns=inference_completed_ns,
                            publish_started_ns=publish_started_ns,
                            publish_completed_ns=publish_completed_ns,
                            worker_apply_completed_ns=time.perf_counter_ns(),
                            source_fallback_count=sum(
                                str(getattr(boundary, "kind", "")) == "fallback"
                                for boundary in boundaries
                            ),
                        )
                    )
                    continue
                expected_bot_count = len(self.controller.bot_ids)
                observation_bot_count = len(getattr(observation, "observations", ()))
                action_bot_count = len(getattr(action, "actions", ())) if action is not None else 0
                if (
                    observation_bot_count != expected_bot_count
                    or action_bot_count != expected_bot_count
                ):
                    # CS:GO can publish one or more live snapshots while fake
                    # clients are still entering/leaving the roster.  Applying
                    # such a partial batch to a fixed training roster would
                    # shift bot identities and either corrupt the trajectory
                    # or raise in SelfPlayController._align_values.  Keep the
                    # game action path alive, but do not train on the batch.
                    self.controller.apply_boundaries(boundaries)
                    metrics_queue.put(
                        {
                            "type": "roster_mismatch",
                            "instance_id": self.controller.instance_id or "default",
                            "match_id": self.controller.match_id or self.controller.episode_id,
                            "policy_generation": self.controller.policy_generation,
                            "server_tick": int(observation.server_tick),
                            "expected_bot_count": expected_bot_count,
                            "observation_bot_count": observation_bot_count,
                            "action_bot_count": action_bot_count,
                        }
                    )
                    if self.controller.status == "finished":
                        stop_event.set()
                        break
                    continue
                self.controller.accept(observation, action, boundaries)
                worker_apply_completed_ns = time.perf_counter_ns()
                for envelope in self.controller.drain_completed(self.rollout_horizon):
                    rollout_queue.put(envelope)
                metrics_queue.put(
                    self._metrics(
                        observation,
                        action,
                        read_started_ns=read_started_ns,
                        read_completed_ns=read_completed_ns,
                        inference_started_ns=inference_started_ns,
                        inference_completed_ns=inference_completed_ns,
                        publish_started_ns=publish_started_ns,
                        publish_completed_ns=publish_completed_ns,
                        worker_apply_completed_ns=worker_apply_completed_ns,
                        source_fallback_count=sum(
                            str(getattr(boundary, "kind", "")) == "fallback"
                            for boundary in boundaries
                        ),
                    )
                )
                if self.controller.status == "finished":
                    stop_event.set()
                    break
        finally:
            pending_controller_boundaries = ()
            for envelope in self.controller.flush_pending(truncated=True):
                rollout_queue.put(envelope)
            if self.controller.match_terminal:
                rollout_queue.put(
                    MatchCompletionV1(
                        instance_id=self.controller.instance_id or "default",
                        match_id=self.controller.match_id or self.controller.episode_id,
                        policy_generation=self.controller.policy_generation,
                        envelopes=self.controller.completed_envelopes,
                    )
                )
            close = getattr(self.transport, "close", None)
            if callable(close):
                close()

    def _activate_staged_policy(self, boundaries: tuple[Any, ...]) -> None:
        activate = getattr(self.runtime, "activate_policy_generation_at_going_live", None)
        if not callable(activate):
            return
        if any(str(getattr(boundary, "kind", "")) == "going_live" for boundary in boundaries):
            activate()

    def _apply_runtime_boundaries(self, boundaries: tuple[Any, ...]) -> None:
        apply_boundaries = getattr(self.runtime, "apply_boundaries", None)
        if callable(apply_boundaries) and boundaries:
            apply_boundaries(boundaries)

    @staticmethod
    def _observation_precedes_round_start(observation: Any, boundaries: tuple[Any, ...]) -> bool:
        for boundary in boundaries:
            if str(getattr(boundary, "kind", "")) != "round_start":
                continue
            try:
                observation_epoch = int(observation.epoch)
                boundary_epoch = int(boundary.epoch)
                observation_tick = int(observation.server_tick)
                boundary_tick = int(boundary.server_tick)
            except (AttributeError, TypeError, ValueError):
                continue
            if observation_epoch != boundary_epoch or observation_tick <= boundary_tick:
                return True
        return False

    def _read_latest_observation(self, observation: Any) -> Any:
        latest = observation
        while True:
            candidate = self.transport.try_read_observation()
            if candidate is None:
                return latest
            latest = candidate

    def _runtime_timing(self) -> dict[str, int]:
        timing = getattr(self.runtime, "last_timing", {})
        if callable(timing):
            timing = timing()
        return dict(timing) if isinstance(timing, dict) else {}

    def _transport_aborted(self) -> bool:
        if bool(getattr(self.transport, "aborted", False)):
            return True
        observation_ring = getattr(self.transport, "observations", None)
        return bool(getattr(observation_ring, "aborted", False))

    def _metrics(
        self,
        observation: Any,
        action: Any,
        *,
        read_started_ns: int,
        read_completed_ns: int,
        inference_started_ns: int,
        inference_completed_ns: int,
        publish_started_ns: int,
        publish_completed_ns: int,
        worker_apply_completed_ns: int,
        source_fallback_count: int = 0,
    ) -> dict[str, Any]:
        source_ack_tick = getattr(self.control_state, "last_latency_tick", None)
        source_ack = source_ack_tick == int(observation.server_tick)
        source_latency_us = (
            getattr(self.control_state, "last_latency_us", None) if source_ack else None
        )
        source_policy_generation = getattr(self.control_state, "policy_generation", None)
        fallback_count = sum(
            item.action_valid_mask == 0 for item in getattr(action, "actions", ())
        )
        runtime_metrics: dict[str, Any] = {}
        metrics_reader = getattr(self.runtime, "metrics", None)
        if callable(metrics_reader):
            try:
                value = metrics_reader()
                if isinstance(value, dict):
                    runtime_metrics = value
            except Exception:
                runtime_metrics = {}
        action_percentiles = runtime_metrics.get("action_ms", {})
        tick_percentiles = runtime_metrics.get("tick_ms", {})
        return {
            "instance_id": self.controller.instance_id or "default",
            "match_id": self.controller.match_id or self.controller.episode_id,
            "policy_generation": self.controller.policy_generation,
            "server_tick": int(observation.server_tick),
            "read_started_ns": read_started_ns,
            "read_completed_ns": read_completed_ns,
            "inference_started_ns": inference_started_ns,
            "inference_completed_ns": inference_completed_ns,
            "publish_started_ns": publish_started_ns,
            "publish_completed_ns": publish_completed_ns,
            "apply_completed_ns": None,
            "worker_apply_completed_ns": worker_apply_completed_ns,
            "read_us": _duration_us(read_started_ns, read_completed_ns),
            "inference_us": _duration_us(inference_started_ns, inference_completed_ns),
            "publish_us": _duration_us(publish_started_ns, publish_completed_ns),
            "apply_us": None,
            "latency_us": source_latency_us,
            "bridge_latency_us": source_latency_us,
            "source_action_apply_ack": source_ack,
            "source_action_apply_tick": source_ack_tick,
            "source_policy_generation": source_policy_generation,
            "rules_validated": "rules_validated" in getattr(self.controller, "boundaries", ()),
            "fallback_count": int(fallback_count),
            "source_fallback_count": int(source_fallback_count),
            "runtime_process_failed": bool(getattr(self.runtime, "process_failed", False)),
            # These snapshots let the capacity probe measure the same worker
            # runtime that serves each server, instead of substituting one
            # parent-side ONNX call for N concurrent IPC workers.
            "runtime_action_p99_ms": float(
                action_percentiles.get("p99", 0.0)
                if isinstance(action_percentiles, dict)
                else 0.0
            ),
            "runtime_tick_p99_ms": float(
                tick_percentiles.get("p99", 0.0)
                if isinstance(tick_percentiles, dict)
                else 0.0
            ),
            "runtime_total_tick_deadline_miss": int(
                runtime_metrics.get("total_tick_deadline_miss", 0)
            ),
            "runtime_permission_violations": int(
                runtime_metrics.get("permission_violations", 0)
            ),
        }


def _duration_us(start_ns: int, end_ns: int) -> float:
    return max(0, int(end_ns) - int(start_ns)) / 1000.0


__all__ = ["MatchCompletionV1", "RolloutEnvelopeV1", "SelfPlayWorker"]

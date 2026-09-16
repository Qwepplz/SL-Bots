from __future__ import annotations

from threading import Event
import time
from typing import Any

from .contracts import Phase
from .get5_control import Get5ControlState
from .runtime import QueueAborted
from .selfplay import RolloutEnvelopeV1, SelfPlayController


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
        try:
            while not stop_event.is_set():
                try:
                    boundaries = self.control_state.consume_available(self.transport)
                    self._activate_staged_policy(boundaries)
                    if not self.transport.wait_for_observation(self.wait_timeout_ms):
                        if self._transport_aborted():
                            break
                        continue
                    read_started_ns = time.perf_counter_ns()
                    observation = self.transport.try_read_observation()
                    read_completed_ns = time.perf_counter_ns()
                except QueueAborted:
                    break
                if observation is None:
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
            for envelope in self.controller.flush_pending(truncated=True):
                rollout_queue.put(envelope)
            close = getattr(self.transport, "close", None)
            if callable(close):
                close()

    def _activate_staged_policy(self, boundaries: tuple[Any, ...]) -> None:
        activate = getattr(self.runtime, "activate_policy_generation_at_going_live", None)
        if not callable(activate):
            return
        if any(str(getattr(boundary, "kind", "")) == "going_live" for boundary in boundaries):
            activate()

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
        }


def _duration_us(start_ns: int, end_ns: int) -> float:
    return max(0, int(end_ns) - int(start_ns)) / 1000.0


__all__ = ["RolloutEnvelopeV1", "SelfPlayWorker"]

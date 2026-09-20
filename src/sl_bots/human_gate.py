"""Human-like hard gate evaluated before any competitive result is read."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import os
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .rewards import human_metric_regression_fraction
from .training_gail import compute_human_like_metrics
from .movement_evaluation import movement_evidence_is_accepted


_MISSING = object()

# This is a contract, not an inferred set.  Allowing the two packages to
# agree on a smaller set would make a missing metric indistinguishable from a
# passing metric and would reopen the gate fail-open path.
REQUIRED_HUMAN_METRICS = (
    "max_angular_velocity_deg_s",
    "max_angular_acceleration_deg_s2",
    "stop_go_ratio",
    "fire_cadence_hz",
    "utility_event_rate",
    "economy_choice_count",
)


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        if name in source:
            return source[name]
        metadata = source.get("metadata")
        if isinstance(metadata, Mapping) and name in metadata:
            return metadata[name]
        return default
    value = getattr(source, name, default)
    if value is not default:
        return value
    metadata = getattr(source, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return default


def _purpose(source: Any) -> str | None:
    value = _value(source, "purpose")
    if hasattr(value, "value"):
        value = value.value
    return None if value is None else str(value)


def _manifest_sha(source: Any) -> str:
    value = _value(source, "sha256", "")
    if value:
        return str(value)
    source_sha = _value(source, "source_sha256", "")
    return str(source_sha or "")


def _records(source: Any, explicit: Iterable[Mapping[str, Any]] | None) -> tuple[Mapping[str, Any], ...]:
    if explicit is not None:
        return tuple(explicit)
    value = _value(source, "records", ())
    if value is None:
        return ()
    return tuple(value)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * ratio


def _validation_quantiles(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        episode_id = str(record.get("episode_id", "all"))
        grouped.setdefault(episode_id, []).append(record)
    samples: dict[str, list[float]] = {}
    for episode_records in grouped.values():
        metrics = asdict(compute_human_like_metrics(episode_records))
        for name in REQUIRED_HUMAN_METRICS:
            if name in metrics:
                samples.setdefault(name, []).append(float(metrics[name]))
    return {
        name: {
            "p05": _percentile(values, 0.05),
            "p95": _percentile(values, 0.95),
        }
        for name, values in samples.items()
    }


def _train_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for record in records:
        for name, value in record.items():
            if isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.setdefault(str(name), []).append(numeric)
    return {name: sum(items) / len(items) for name, items in values.items() if items}


@dataclass(frozen=True)
class HumanBaselineV1:
    train_statistics: Mapping[str, float]
    validation_quantiles: Mapping[str, Mapping[str, float]]
    train_manifest_sha256: str
    validation_manifest_sha256: str
    validation_intent_nll: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_statistics", dict(self.train_statistics))
        object.__setattr__(
            self,
            "validation_quantiles",
            {name: dict(values) for name, values in self.validation_quantiles.items()},
        )
        if self.validation_intent_nll < 0.0 or not math.isfinite(self.validation_intent_nll):
            raise ValueError("validation intent NLL must be finite and non-negative")


@dataclass(frozen=True)
class HumanGateResultV1:
    accepted: bool
    reasons: tuple[str, ...]
    evidence: Mapping[str, Any]
    baseline_sha256: str
    candidate_sha256: str
    pending_pointer_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "evidence", dict(self.evidence))


def build_human_baseline(
    train_manifest: Any,
    validation_manifest: Any,
    *,
    train_records: Iterable[Mapping[str, Any]] | None = None,
    validation_records: Iterable[Mapping[str, Any]] | None = None,
) -> HumanBaselineV1:
    """Fit descriptive statistics from train and gates from validation only."""

    for manifest in (train_manifest, validation_manifest):
        purpose = _purpose(manifest)
        if purpose is not None and purpose != "test_only":
            raise ValueError("human-like test gate requires test_only manifests")
    train_values = _records(train_manifest, train_records)
    validation_values = _records(validation_manifest, validation_records)
    raw_intent_nll = _value(validation_manifest, "intent_nll", _MISSING)
    if raw_intent_nll is _MISSING:
        raise ValueError("validation manifest must provide intent_nll for the human gate")
    intent_nll = float(raw_intent_nll)
    return HumanBaselineV1(
        train_statistics=_train_statistics(train_values),
        validation_quantiles=_validation_quantiles(validation_values),
        train_manifest_sha256=_manifest_sha(train_manifest),
        validation_manifest_sha256=_manifest_sha(validation_manifest),
        validation_intent_nll=intent_nll,
    )


def _package_sha256(package: Any) -> str:
    value = _value(package, "sha256", "")
    if value:
        return str(value)
    decision = _value(package, "decision_sha256", "")
    action = _value(package, "action_sha256", "")
    if decision or action:
        return hashlib.sha256(f"{decision}:{action}".encode("utf-8")).hexdigest()
    return ""


def _evidence_value(source: Any, name: str, default: Any = _MISSING) -> Any:
    """Read gate evidence from an object or its serialized package metadata."""

    value = _value(source, name, _MISSING)
    if value is not _MISSING:
        return value
    metadata = _value(source, "metadata", {})
    if not isinstance(metadata, Mapping):
        return default
    if name in metadata:
        return metadata[name]
    metrics = metadata.get("metrics")
    if isinstance(metrics, Mapping) and name in metrics:
        return metrics[name]
    return default


def _package_metrics(package: Any) -> dict[str, float]:
    value = _evidence_value(package, "human_metrics", _MISSING)
    if value is _MISSING:
        value = _evidence_value(package, "metrics", {})
    if isinstance(value, Mapping) and isinstance(value.get("human_metrics"), Mapping):
        value = value["human_metrics"]
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for name, item in value.items():
        if isinstance(item, bool) or isinstance(item, Mapping):
            continue
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result[str(name)] = numeric
    return result


def _package_movement_evidence(package: Any) -> Mapping[str, Any] | None:
    """Read optional v3 movement evidence without treating it as legacy metrics."""

    for name in ("movement_evidence", "movement_gate", "movement_metrics"):
        value = _evidence_value(package, name, _MISSING)
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _isolate_pending_pointer(path: Path) -> Path | None:
    if not path.is_file():
        return None
    rejected = path.with_name(f"{path.stem}.rejected{path.suffix}")
    os.replace(path, rejected)
    return rejected


def evaluate_human_gate(
    baseline_package: Any,
    candidate_package: Any,
    validation_manifest: Any,
    *,
    pending_pointer: str | Path | None = None,
    active_pointer: str | Path | None = None,
) -> HumanGateResultV1:
    """Evaluate all human-like gates before any match score/winner is available."""

    reasons: list[str] = []
    baseline_sha = _package_sha256(baseline_package)
    candidate_sha = _package_sha256(candidate_package)
    raw_baseline_nll = _evidence_value(baseline_package, "intent_nll", _MISSING)
    raw_candidate_nll = _evidence_value(candidate_package, "intent_nll", _MISSING)
    baseline_nll: float | None = None
    candidate_nll: float | None = None
    nll_limit: float | None = None
    if raw_baseline_nll is _MISSING or raw_candidate_nll is _MISSING:
        reasons.append("intent NLL evidence is missing for baseline or candidate")
    else:
        try:
            baseline_nll = float(raw_baseline_nll)
            candidate_nll = float(raw_candidate_nll)
        except (TypeError, ValueError):
            reasons.append("intent NLL evidence is not numeric")
        else:
            if not all(math.isfinite(value) and value >= 0.0 for value in (baseline_nll, candidate_nll)):
                reasons.append("intent NLL evidence is not finite and non-negative")
            else:
                nll_limit = baseline_nll * 1.05
                if candidate_nll > nll_limit:
                    reasons.append(
                        f"intent NLL worsened by more than 5% ({candidate_nll:.6g} > {nll_limit:.6g})"
                    )

    baseline_metrics = _package_metrics(baseline_package)
    candidate_metrics = _package_metrics(candidate_package)
    baseline_movement = _package_movement_evidence(baseline_package)
    candidate_movement = _package_movement_evidence(candidate_package)
    for package_name, movement_evidence in (
        ("baseline", baseline_movement),
        ("candidate", candidate_movement),
    ):
        movement_status = movement_evidence_is_accepted(movement_evidence)
        if movement_status is False:
            reasons.append(f"movement v3 gate rejected {package_name} evidence")
    required_metrics = set(REQUIRED_HUMAN_METRICS)
    for package_name, metrics in (("baseline", baseline_metrics), ("candidate", candidate_metrics)):
        missing = sorted(required_metrics - set(metrics))
        extra = sorted(set(metrics) - required_metrics)
        if missing or extra:
            reasons.append(
                f"required human metrics set is incomplete for {package_name} "
                f"(missing={missing}, extra={extra})"
            )
    if set(baseline_metrics) != set(candidate_metrics):
        missing_from_candidate = sorted(set(baseline_metrics) - set(candidate_metrics))
        missing_from_baseline = sorted(set(candidate_metrics) - set(baseline_metrics))
        reasons.append(
            "human metric set differs between baseline and candidate "
            f"(missing_from_candidate={missing_from_candidate}, "
            f"missing_from_baseline={missing_from_baseline})"
        )
    quantiles = _value(validation_manifest, "quantiles", None)
    if not isinstance(quantiles, Mapping):
        quantiles = _evidence_value(baseline_package, "validation_quantiles", {})
    if not isinstance(quantiles, Mapping) or not quantiles:
        reasons.append("validation quantiles evidence is missing")
        quantiles = {}
    quantile_evidence: dict[str, Any] = {}
    expected_metric_names = sorted(required_metrics)
    for name in expected_metric_names:
        value = candidate_metrics.get(name)
        threshold = quantiles.get(name) if isinstance(quantiles, Mapping) else None
        if not isinstance(threshold, Mapping):
            reasons.append(f"validation quantile is missing for human metric {name}")
            continue
        raw_lower = threshold.get("p05", threshold.get("p5", _MISSING))
        raw_upper = threshold.get("p95", _MISSING)
        try:
            lower = float(raw_lower)
            upper = float(raw_upper)
        except (TypeError, ValueError):
            reasons.append(f"validation quantile bounds are incomplete for human metric {name}")
            continue
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            reasons.append(f"validation quantile bounds are invalid for human metric {name}")
            continue
        if value is None:
            continue
        quantile_evidence[name] = {"value": value, "p05": lower, "p95": upper}
        if value < lower or value > upper:
            reasons.append(
                f"{name} is outside validation 5%/95% quantile ({value:.6g} not in [{lower:.6g}, {upper:.6g}])"
            )

    regressions = human_metric_regression_fraction(baseline_metrics, candidate_metrics)
    for name, fraction in regressions.items():
        if fraction > 0.10:
            reasons.append(f"{name} regressed by more than 10% ({fraction * 100:.3g}%)")

    evidence = {
        "intent_nll": {"baseline": baseline_nll, "candidate": candidate_nll, "limit": nll_limit},
        "validation_quantiles": quantile_evidence,
        "metric_regression_fraction": regressions,
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "active_pointer": None if active_pointer is None else str(active_pointer),
        "movement": {
            "baseline": baseline_movement,
            "candidate": candidate_movement,
        },
    }
    rejected_path: Path | None = None
    if reasons and pending_pointer is not None:
        rejected_path = _isolate_pending_pointer(Path(pending_pointer))
    return HumanGateResultV1(
        accepted=not reasons,
        reasons=tuple(reasons),
        evidence=evidence,
        baseline_sha256=baseline_sha,
        candidate_sha256=candidate_sha,
        pending_pointer_path=None if rejected_path is None else str(rejected_path),
    )

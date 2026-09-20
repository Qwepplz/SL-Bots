"""Offline movement metrics, ablation manifests, and test-split audits.

The evaluator deliberately operates on already materialized rows.  It never opens a
Demo, a Parquet shard, or a server.  This keeps the metric implementation usable in
the small pilot and makes the validation/test boundary explicit to callers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .movement import (
    IDLE_CLASS,
    JUMP_NO,
    MOVEMENT_CLASS_COUNT,
    MovementLabelV1,
    STANCE_RUN,
    label_from_action,
)


ABLATION_IDS = ("A", "B", "C", "D", "E")
MOVEMENT_EVALUATION_SCHEMA = "movement-evaluation-v1"
MOVEMENT_ABLATION_SCHEMA = "movement-ablation-v1"
FORMAL_MOVEMENT_GATE_SEEDS = (7, 17, 29)
_HORIZON_NAMES = ("horizon_0", "horizon_1", "horizon_2")
_SLICE_FIELDS = ("demo", "team", "side", "phase", "seen_enemy", "action_class")
MOVEMENT_MIRROR_PERMUTATION = (0, 1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2)
_MISSING = object()


def _plain(value: Any) -> Any:
    """Convert common tensor/scalar containers without requiring NumPy."""

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _rows(value: Any) -> list[list[float]]:
    value = _plain(value)
    if not _is_sequence(value):
        return [[float(value)]]
    values = list(value)
    if not values:
        return []
    if _is_sequence(values[0]):
        return [[float(item) for item in row] for row in values]
    return [[float(item) for item in values]]


def _button_bits_from_logits(values: Sequence[float], threshold: float = 0.5) -> int:
    bits = 0
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        probability = 1.0 / (1.0 + math.exp(-value))
        if probability > threshold:
            bits |= 1 << index
    return bits


def _button_value(buttons: Any, index: int) -> int:
    buttons = _plain(buttons)
    if buttons is None:
        return 0
    if isinstance(buttons, bool):
        raise TypeError("buttons must be an integer bit mask or button logits")
    if isinstance(buttons, int):
        return buttons
    if _is_sequence(buttons):
        values = list(buttons)
        if values and _is_sequence(values[0]):
            values = list(values[min(index, len(values) - 1)])
        if values and all(isinstance(item, int) and not isinstance(item, bool) for item in values):
            return int(values[min(index, len(values) - 1)]) if len(values) <= 3 else _button_bits_from_logits(values)
        return _button_bits_from_logits([float(item) for item in values])
    raise TypeError("buttons must be an integer bit mask or button logits")


def adapt_v2_beta_to_movement_labels(
    movement_alpha: Any,
    movement_beta: Any | None = None,
    *,
    buttons: Any = None,
) -> MovementLabelV1 | tuple[MovementLabelV1, ...]:
    """Adapt v2 Beta means to the shared discrete movement label space.

    v2 predicts one current movement vector.  Consequently only horizon zero is
    marked valid; marking the two future horizons valid would manufacture future
    evidence and make the v2 baseline incomparable with v3.

    ``movement_alpha`` may also be a v2 ``PolicyOutputV1``-like object or mapping
    containing ``movement_alpha``/``movement_beta`` and optional ``button_logits``.
    """

    source = movement_alpha
    if movement_beta is None:
        if isinstance(source, Mapping):
            movement_beta = source.get("movement_beta")
            movement_alpha = source.get("movement_alpha")
            if buttons is None:
                buttons = source.get("buttons", source.get("button_logits"))
        else:
            movement_beta = getattr(source, "movement_beta", None)
            movement_alpha = getattr(source, "movement_alpha", None)
            if buttons is None:
                buttons = getattr(source, "buttons", getattr(source, "button_logits", None))
    if movement_alpha is None or movement_beta is None:
        raise ValueError("v2 Beta output must provide movement_alpha and movement_beta")
    alpha_rows = _rows(movement_alpha)
    beta_rows = _rows(movement_beta)
    if len(alpha_rows) != len(beta_rows) or not alpha_rows:
        raise ValueError("movement_alpha and movement_beta must have matching non-empty batches")
    labels: list[MovementLabelV1] = []
    for row_index, (alpha_row, beta_row) in enumerate(zip(alpha_rows, beta_rows, strict=True)):
        if len(alpha_row) != 3 or len(beta_row) != 3:
            raise ValueError("v2 movement Beta parameters must contain exactly three axes")
        means = []
        for alpha, beta in zip(alpha_row, beta_row, strict=True):
            if alpha <= 0.0 or beta <= 0.0 or not math.isfinite(alpha + beta):
                raise ValueError("v2 movement Beta parameters must be finite and positive")
            means.append(2.0 * alpha / (alpha + beta) - 1.0)
        current = label_from_action(
            forward=means[0],
            side=means[1],
            up=means[2],
            buttons=_button_value(buttons, row_index),
        )
        labels.append(
            MovementLabelV1(
                move=(current.move[0], IDLE_CLASS, IDLE_CLASS),
                stance=(current.stance[0], STANCE_RUN, STANCE_RUN),
                jump=(current.jump[0], JUMP_NO, JUMP_NO),
                valid=(True, False, False),
                stance_valid=(current.stance_valid[0], False, False),
            )
        )
    return labels[0] if len(labels) == 1 else tuple(labels)


# Short aliases keep the adapter discoverable from either terminology used in the
# v2 code and in the redesign document.
adapt_beta_movement_to_labels = adapt_v2_beta_to_movement_labels
unified_movement_labels_from_v2 = adapt_v2_beta_to_movement_labels


def _mask_values(values: Sequence[Any], mask: Sequence[bool] | None) -> list[Any]:
    if mask is None:
        return list(values)
    if len(mask) != len(values):
        raise ValueError("mask length must match values")
    return [value for value, include in zip(values, mask, strict=True) if bool(include)]


def circular_direction_error(
    predicted: Sequence[int],
    target: Sequence[int],
    *,
    mask: Sequence[bool] | None = None,
    direction_count: int = 16,
) -> float:
    """Mean circular error in degrees; idle-vs-moving is a 180° error."""

    if direction_count <= 0:
        raise ValueError("direction_count must be positive")
    if len(predicted) != len(target):
        raise ValueError("predicted and target lengths must match")
    if mask is not None and len(mask) != len(predicted):
        raise ValueError("mask length must match values")
    errors: list[float] = []
    sector = 360.0 / float(direction_count)
    for index, (raw_predicted, raw_target) in enumerate(zip(predicted, target, strict=True)):
        if mask is not None and not bool(mask[index]):
            continue
        predicted_class = int(raw_predicted)
        target_class = int(raw_target)
        if predicted_class == IDLE_CLASS and target_class == IDLE_CLASS:
            errors.append(0.0)
        elif predicted_class == IDLE_CLASS or target_class == IDLE_CLASS:
            errors.append(180.0)
        else:
            predicted_angle = (predicted_class - 1) * sector
            target_angle = (target_class - 1) * sector
            errors.append(abs((predicted_angle - target_angle + 180.0) % 360.0 - 180.0))
    return sum(errors) / len(errors) if errors else 0.0


def macro_f1(
    predicted: Sequence[int],
    target: Sequence[int],
    *,
    mask: Sequence[bool] | None = None,
    class_count: int | None = None,
) -> float:
    if len(predicted) != len(target):
        raise ValueError("predicted and target lengths must match")
    if mask is not None and len(mask) != len(predicted):
        raise ValueError("mask length must match values")
    pairs = [
        (int(p), int(t))
        for index, (p, t) in enumerate(zip(predicted, target, strict=True))
        if mask is None or bool(mask[index])
    ]
    if not pairs:
        return 0.0
    classes = set(range(class_count)) if class_count is not None else set()
    classes.update(p for p, _ in pairs)
    classes.update(t for _, t in pairs)
    scores: list[float] = []
    for class_id in sorted(classes):
        true_positive = sum(p == class_id and t == class_id for p, t in pairs)
        false_positive = sum(p == class_id and t != class_id for p, t in pairs)
        false_negative = sum(p != class_id and t == class_id for p, t in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return sum(scores) / len(scores)


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    target: Sequence[int],
    *,
    bins: int = 10,
    mask: Sequence[bool] | None = None,
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if len(probabilities) != len(target):
        raise ValueError("probabilities and target lengths must match")
    if mask is not None and len(mask) != len(target):
        raise ValueError("mask length must match values")
    bucket: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for index, (row, raw_target) in enumerate(zip(probabilities, target, strict=True)):
        if mask is not None and not bool(mask[index]):
            continue
        values = [float(item) for item in row]
        if not values:
            raise ValueError("probability rows must not be empty")
        predicted = max(range(len(values)), key=values.__getitem__)
        confidence = max(values)
        bucket_index = min(bins - 1, max(0, int(confidence * bins)))
        bucket[bucket_index].append((confidence, predicted == int(raw_target)))
    total = sum(len(items) for items in bucket)
    if not total:
        return 0.0
    return sum(
        len(items) / total
        * abs(
            sum(correct for _, correct in items) / len(items)
            - sum(confidence for confidence, _ in items) / len(items)
        )
        for items in bucket
        if items
    )


def brier_score(
    probabilities: Sequence[Sequence[float]],
    target: Sequence[int],
    *,
    mask: Sequence[bool] | None = None,
) -> float:
    if len(probabilities) != len(target):
        raise ValueError("probabilities and target lengths must match")
    if mask is not None and len(mask) != len(target):
        raise ValueError("mask length must match values")
    scores: list[float] = []
    for index, (row, raw_target) in enumerate(zip(probabilities, target, strict=True)):
        if mask is not None and not bool(mask[index]):
            continue
        values = [float(item) for item in row]
        target_index = int(raw_target)
        if not 0 <= target_index < len(values):
            raise ValueError("target class is outside probability row")
        scores.append(sum((value - (1.0 if class_id == target_index else 0.0)) ** 2 for class_id, value in enumerate(values)))
    return sum(scores) / len(scores) if scores else 0.0


def _normalized_distribution(values: Sequence[float]) -> list[float]:
    numbers = [float(value) for value in values]
    if any(value < 0.0 or not math.isfinite(value) for value in numbers):
        raise ValueError("distributions must contain finite non-negative values")
    total = sum(numbers)
    if total <= 0.0:
        raise ValueError("distribution must contain positive mass")
    return [value / total for value in numbers]


def js_divergence(predicted: Sequence[float], target: Sequence[float]) -> float:
    if len(predicted) != len(target) or not predicted:
        raise ValueError("distributions must have equal non-zero length")
    p = _normalized_distribution(predicted)
    q = _normalized_distribution(target)
    midpoint = [(left + right) / 2.0 for left, right in zip(p, q, strict=True)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(value * math.log(value / other) for value, other in zip(left, right, strict=True) if value > 0.0)

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


def mean_displacement_error(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    mask: Sequence[bool] | None = None,
) -> float:
    if len(predicted) != len(target):
        raise ValueError("predicted and target positions must match")
    if mask is not None and len(mask) != len(predicted):
        raise ValueError("mask length must match values")
    distances: list[float] = []
    for index, (left, right) in enumerate(zip(predicted, target, strict=True)):
        if mask is not None and not bool(mask[index]):
            continue
        if len(left) != len(right) or not left:
            raise ValueError("position vectors must have equal non-zero length")
        distances.append(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))))
    return sum(distances) / len(distances) if distances else 0.0


def mirror_equivariance_error(
    probabilities: Sequence[Sequence[float]],
    mirrored_probabilities: Sequence[Sequence[float]],
    *,
    direction_permutation: Sequence[int] | None = None,
) -> float:
    if len(probabilities) != len(mirrored_probabilities):
        raise ValueError("probability batches must match")
    errors: list[float] = []
    for original, mirrored in zip(probabilities, mirrored_probabilities, strict=True):
        left = [float(item) for item in original]
        right = [float(item) for item in mirrored]
        if len(left) != len(right):
            raise ValueError("mirrored probability rows must match")
        if direction_permutation is None and len(left) == MOVEMENT_CLASS_COUNT:
            permutation = list(MOVEMENT_MIRROR_PERMUTATION)
        else:
            permutation = list(direction_permutation or range(len(left)))
        if len(permutation) != len(left) or sorted(permutation) != list(range(len(left))):
            raise ValueError("direction_permutation must be a permutation of probability columns")
        expected = [0.0] * len(left)
        for source_index, destination_index in enumerate(permutation):
            expected[destination_index] = left[source_index]
        errors.append(sum(abs(a - b) for a, b in zip(expected, right, strict=True)) / len(left))
    return sum(errors) / len(errors) if errors else 0.0


# Common metric spellings used by reports and downstream callers.
ece = expected_calibration_error
displacement_error = mean_displacement_error
mirror_error = mirror_equivariance_error


def _read_member(source: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _container(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name, _MISSING)
        if value is not _MISSING:
            return value
    return row


def _field(row: Mapping[str, Any], side: str, name: str) -> Any:
    nested = _container(
        row,
        ("prediction", "predicted", "pred", "output") if side == "pred" else ("target", "label", "truth"),
    )
    value = _read_member(nested, name, _MISSING)
    if value is not _MISSING:
        return value
    for key in (
        f"{side}_{name}",
        f"{side}icted_{name}" if side == "pred" else f"target_{name}",
        f"{side}iction_{name}" if side == "pred" else f"truth_{name}",
    ):
        if key in row:
            return row[key]
    return _MISSING


def _horizon_value(value: Any, horizon: int, *, probability: bool = False) -> Any:
    value = _plain(value)
    if not _is_sequence(value):
        return value
    values = list(value)
    if not values:
        return _MISSING
    if _is_sequence(values[0]):
        return values[horizon] if horizon < len(values) else _MISSING
    if not probability and len(values) == 3:
        return values[horizon]
    return values


def _int_value(value: Any) -> int | None:
    if value is _MISSING or value is None:
        return None
    value = _plain(value)
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _probability_row(row: Mapping[str, Any], horizon: int) -> list[float] | None:
    value = _field(row, "pred", "move_probs")
    if value is _MISSING:
        value = _field(row, "pred", "movement_probabilities")
    if value is _MISSING:
        return None
    value = _horizon_value(value, horizon, probability=True)
    if value is _MISSING or not _is_sequence(value):
        return None
    return [float(item) for item in value]


def _target_position(row: Mapping[str, Any], side: str, horizon: int) -> Sequence[float] | None:
    for name in ("position", "xy", "pos"):
        value = _field(row, side, name)
        if value is not _MISSING:
            value = _horizon_value(value, horizon)
            if _is_sequence(value):
                return value
    return None


def _entropy(values: Sequence[float]) -> float:
    total = sum(max(0.0, float(item)) for item in values)
    if total <= 0.0:
        return 0.0
    return -sum((float(item) / total) * math.log(float(item) / total) for item in values if float(item) > 0.0)


def _negative_log_likelihood(probabilities: Sequence[Sequence[float]], target: Sequence[int]) -> float:
    if len(probabilities) != len(target):
        raise ValueError("probabilities and target lengths must match")
    values: list[float] = []
    for row, raw_target in zip(probabilities, target, strict=True):
        distribution = _normalized_distribution(row)
        target_index = int(raw_target)
        if not 0 <= target_index < len(distribution):
            raise ValueError("target class is outside probability row")
        values.append(-math.log(max(distribution[target_index], 1e-12)))
    return sum(values) / len(values) if values else 0.0


def _binary_scores(predicted: Sequence[int], target: Sequence[int]) -> tuple[float, float, float]:
    true_positive = sum(p == 1 and t == 1 for p, t in zip(predicted, target, strict=True))
    false_positive = sum(p == 1 and t == 0 for p, t in zip(predicted, target, strict=True))
    false_negative = sum(p == 0 and t == 1 for p, t in zip(predicted, target, strict=True))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0, "n": 0}
    z = 1.96
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(fraction * (1.0 - fraction) / total + z * z / (4.0 * total * total)) / denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin), "n": total}


def _metric_rows(rows: Sequence[Mapping[str, Any]], horizon: int) -> tuple[dict[str, Any], int, dict[str, Any], tuple[str, ...]]:
    predicted_move: list[int] = []
    target_move: list[int] = []
    predicted_stance: list[int] = []
    target_stance: list[int] = []
    predicted_jump: list[int] = []
    target_jump: list[int] = []
    probabilities: list[list[float]] = []
    positions_predicted: list[Sequence[float]] = []
    positions_target: list[Sequence[float]] = []
    mirror_pairs: list[tuple[list[float], list[float]]] = []
    for row in rows:
        target_valid = _field(row, "target", "valid")
        if target_valid is _MISSING:
            target_valid = row.get("valid", True)
        target_valid = _horizon_value(target_valid, horizon)
        if target_valid is _MISSING or not bool(target_valid):
            continue
        predicted = _int_value(_horizon_value(_field(row, "pred", "move"), horizon))
        target = _int_value(_horizon_value(_field(row, "target", "move"), horizon))
        if predicted is None or target is None:
            continue
        predicted_move.append(predicted)
        target_move.append(target)
        predicted_stance_value = _int_value(_horizon_value(_field(row, "pred", "stance"), horizon))
        target_stance_value = _int_value(_horizon_value(_field(row, "target", "stance"), horizon))
        stance_valid = _field(row, "target", "stance_valid")
        if stance_valid is _MISSING:
            stance_valid = row.get("stance_valid", target != IDLE_CLASS)
        stance_valid = bool(_horizon_value(stance_valid, horizon))
        if predicted_stance_value is not None and target_stance_value is not None and stance_valid:
            predicted_stance.append(predicted_stance_value)
            target_stance.append(target_stance_value)
        predicted_jump_value = _int_value(_horizon_value(_field(row, "pred", "jump"), horizon))
        target_jump_value = _int_value(_horizon_value(_field(row, "target", "jump"), horizon))
        if predicted_jump_value is not None and target_jump_value is not None:
            predicted_jump.append(predicted_jump_value)
            target_jump.append(target_jump_value)
        probability_row = _probability_row(row, horizon)
        if probability_row is None:
            probability_row = [1.0 if index == predicted else 0.0 for index in range(MOVEMENT_CLASS_COUNT)]
        probabilities.append(probability_row)
        predicted_position = _target_position(row, "pred", horizon)
        target_position = _target_position(row, "target", horizon)
        if predicted_position is not None and target_position is not None:
            positions_predicted.append(predicted_position)
            positions_target.append(target_position)
        mirror_value = _field(row, "pred", "mirrored_move_probs")
        if mirror_value is not _MISSING:
            mirror_value = _horizon_value(mirror_value, horizon, probability=True)
            if _is_sequence(mirror_value):
                mirror_pairs.append((probability_row, [float(item) for item in mirror_value]))
    count = len(predicted_move)
    metrics: dict[str, Any] = {
        "move_circular_mae_deg": circular_direction_error(predicted_move, target_move),
        "move_macro_f1": macro_f1(predicted_move, target_move, class_count=MOVEMENT_CLASS_COUNT),
        "move_top1": sum(p == t for p, t in zip(predicted_move, target_move, strict=True)) / count if count else 0.0,
        "move_top3": 0.0,
        "move_balanced_accuracy": 0.0,
        "stance_macro_f1": macro_f1(predicted_stance, target_stance, class_count=3),
        "jump_precision": 0.0,
        "jump_recall": 0.0,
        "jump_f1": 0.0,
        "ece": expected_calibration_error(probabilities, target_move) if count else 0.0,
        "brier": brier_score(probabilities, target_move) if count else 0.0,
        "move_nll": _negative_log_likelihood(probabilities, target_move) if count else 0.0,
        "prediction_entropy": sum(_entropy(row) for row in probabilities) / len(probabilities) if probabilities else 0.0,
        "distribution_js": 0.0,
        "predicted_move_class_count": len(set(predicted_move)),
        "displacement_error": mean_displacement_error(positions_predicted, positions_target) if positions_predicted else 0.0,
        "mirror_equivariance_error": mirror_equivariance_error(
            [pair[0] for pair in mirror_pairs],
            [pair[1] for pair in mirror_pairs],
        ) if mirror_pairs else None,
        "stance_distribution_js": 0.0,
        "jump_distribution_js": 0.0,
        "stance_positive_count": sum(value != STANCE_RUN for value in target_stance),
        "jump_positive_count": sum(value == 1 for value in target_jump),
    }
    if count:
        top3 = sum(target in sorted(range(len(row)), key=row.__getitem__, reverse=True)[:3] for row, target in zip(probabilities, target_move, strict=True))
        metrics["move_top3"] = top3 / count
        recalls = []
        for class_id in sorted(set(target_move)):
            class_total = sum(target == class_id for target in target_move)
            recalls.append(sum(p == t == class_id for p, t in zip(predicted_move, target_move, strict=True)) / class_total)
        metrics["move_balanced_accuracy"] = sum(recalls) / len(recalls)
        predicted_distribution = [predicted_move.count(class_id) for class_id in range(MOVEMENT_CLASS_COUNT)]
        target_distribution = [target_move.count(class_id) for class_id in range(MOVEMENT_CLASS_COUNT)]
        metrics["distribution_js"] = js_divergence(predicted_distribution, target_distribution)
    if predicted_stance:
        metrics["stance_distribution_js"] = js_divergence(
            [predicted_stance.count(class_id) for class_id in range(3)],
            [target_stance.count(class_id) for class_id in range(3)],
        )
    if predicted_jump:
        precision, recall, f1 = _binary_scores(predicted_jump, target_jump)
        metrics["jump_precision"] = precision
        metrics["jump_recall"] = recall
        metrics["jump_f1"] = f1
        metrics["jump_macro_f1"] = macro_f1(predicted_jump, target_jump, class_count=2)
        metrics["jump_distribution_js"] = js_divergence(
            [predicted_jump.count(class_id) for class_id in range(2)],
            [target_jump.count(class_id) for class_id in range(2)],
        )
    confusion: dict[str, dict[str, int]] = {}
    for predicted, target in zip(predicted_stance, target_stance, strict=True):
        confusion.setdefault(str(target), {}).setdefault(str(predicted), 0)
        confusion[str(target)][str(predicted)] += 1
    metrics["stance_confusion_matrix"] = confusion
    confidence_intervals = {
        "move_top1": _wilson(sum(p == t for p, t in zip(predicted_move, target_move, strict=True)), count),
        "jump_f1": {"low": max(0.0, float(metrics["jump_f1"]) - 1.96 / math.sqrt(max(1, len(predicted_jump)))), "high": min(1.0, float(metrics["jump_f1"]) + 1.96 / math.sqrt(max(1, len(predicted_jump)))), "n": len(predicted_jump)},
    }
    indeterminate: list[str] = []
    if not count:
        indeterminate.append("no valid movement labels")
    if not positions_predicted:
        indeterminate.append("decoded displacement evidence is missing")
    if not mirror_pairs:
        indeterminate.append("mirror-equivalence evidence is missing")
    return metrics, count, confidence_intervals, tuple(indeterminate)


@dataclass(frozen=True)
class MovementEvaluationReportV1:
    metrics: Mapping[str, Mapping[str, Any]]
    raw_counts: Mapping[str, int]
    confidence_intervals: Mapping[str, Mapping[str, Any]]
    slices: Mapping[str, Mapping[str, Any]]
    worst_slice: Mapping[str, Any]
    indeterminate: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", {str(key): dict(value) for key, value in self.metrics.items()})
        object.__setattr__(self, "raw_counts", {str(key): int(value) for key, value in self.raw_counts.items()})
        object.__setattr__(self, "confidence_intervals", {str(key): dict(value) for key, value in self.confidence_intervals.items()})
        object.__setattr__(self, "slices", {str(key): dict(value) for key, value in self.slices.items()})
        object.__setattr__(self, "worst_slice", dict(self.worst_slice))
        object.__setattr__(self, "indeterminate", tuple(str(item) for item in self.indeterminate))
        object.__setattr__(self, "failures", tuple(str(item) for item in self.failures))

    @property
    def accepted(self) -> bool:
        return not self.indeterminate and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MOVEMENT_EVALUATION_SCHEMA,
            "metrics": self.metrics,
            "raw_counts": self.raw_counts,
            "confidence_intervals": self.confidence_intervals,
            "slices": self.slices,
            "worst_slice": self.worst_slice,
            "indeterminate": list(self.indeterminate),
            "failures": list(self.failures),
            "accepted": self.accepted,
        }


def _slice_value(row: Mapping[str, Any], field: str) -> Any:
    aliases = {
        "demo": ("demo", "demo_id", "demo_sha256"),
        "team": ("team", "team_id"),
        "side": ("side", "team_side"),
        "phase": ("phase", "round_phase"),
        "seen_enemy": ("seen_enemy", "enemy_visible", "can_see_enemy"),
        "action_class": ("action_class", "movement_class", "coarse_action_class"),
    }
    for name in aliases[field]:
        if name in row:
            return row[name]
    return _MISSING


def evaluate_movement_predictions(
    records: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = _SLICE_FIELDS,
) -> MovementEvaluationReportV1:
    rows = tuple(records)
    metrics: dict[str, Mapping[str, Any]] = {}
    raw_counts: dict[str, int] = {}
    confidence_intervals: dict[str, Mapping[str, Any]] = {}
    all_indeterminate: list[str] = []
    for horizon, horizon_name in enumerate(_HORIZON_NAMES):
        values, count, intervals, indeterminate = _metric_rows(rows, horizon)
        metrics[horizon_name] = values
        raw_counts[horizon_name] = count
        confidence_intervals[horizon_name] = intervals
        all_indeterminate.extend(f"{horizon_name}: {item}" for item in indeterminate)
    slices: dict[str, Mapping[str, Any]] = {}
    slice_candidates: list[tuple[str, Mapping[str, Any]]] = []
    for field in group_fields:
        if field not in _SLICE_FIELDS:
            raise ValueError(f"unsupported movement slice field: {field}")
        values = sorted({str(_slice_value(row, field)) for row in rows if _slice_value(row, field) is not _MISSING})
        for value in values:
            subset = tuple(row for row in rows if str(_slice_value(row, field)) == value)
            summary, count, _, _ = _metric_rows(subset, 0)
            key = f"{field}={value}"
            payload = {"raw_count": count, "metrics": summary}
            slices[key] = payload
            slice_candidates.append((key, payload))
    full_group_fields = tuple(
        field for field in group_fields if any(_slice_value(row, field) is not _MISSING for row in rows)
    )
    if full_group_fields:
        full_group_rows = [row for row in rows if all(_slice_value(row, field) is not _MISSING for field in full_group_fields)]
        full_keys = sorted({"|".join(f"{field}={_slice_value(row, field)}" for field in full_group_fields) for row in full_group_rows})
        for key in full_keys:
            expected = dict(item.split("=", 1) for item in key.split("|"))
            subset = tuple(row for row in full_group_rows if all(str(_slice_value(row, field)) == expected[field] for field in full_group_fields))
            summary, count, _, _ = _metric_rows(subset, 0)
            payload = {"raw_count": count, "metrics": summary}
            slices[key] = payload
            slice_candidates.append((key, payload))
    if slice_candidates:
        full_slice_candidates = [item for item in slice_candidates if "|" in item[0]]
        candidates = full_slice_candidates or slice_candidates
        worst_key, worst_payload = min(
            candidates,
            key=lambda item: (float(item[1]["metrics"].get("move_macro_f1", 0.0)), -int(item[1]["raw_count"]), item[0]),
        )
        worst_slice = {"key": worst_key, **worst_payload}
    else:
        worst_slice = {}
    return MovementEvaluationReportV1(
        metrics=metrics,
        raw_counts=raw_counts,
        confidence_intervals=confidence_intervals,
        slices=slices,
        worst_slice=worst_slice,
        indeterminate=tuple(dict.fromkeys(all_indeterminate)),
    )


@dataclass(frozen=True)
class MovementGateReportV1:
    accepted: bool
    raw_counts: Mapping[str, int]
    confidence_intervals: Mapping[str, Mapping[str, Any]]
    indeterminate: tuple[str, ...]
    failures: tuple[str, ...]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "movement-gate-v1",
            "accepted": self.accepted,
            "raw_counts": dict(self.raw_counts),
            "confidence_intervals": dict(self.confidence_intervals),
            "indeterminate": list(self.indeterminate),
            "failures": list(self.failures),
            "evidence": dict(self.evidence),
        }


def _gate_metric_value(value: Any) -> tuple[float | None, str | None]:
    if value is _MISSING or value is None:
        return None, "missing"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, "non-numeric"
    if not math.isfinite(numeric):
        return None, "not finite"
    return numeric, None


def build_movement_gate_report(
    candidate: MovementEvaluationReportV1,
    *,
    baseline: MovementEvaluationReportV1 | None = None,
    train_prior_baseline: MovementEvaluationReportV1 | None = None,
    architecture_baseline: MovementEvaluationReportV1 | None = None,
    minimum_count: int = 1,
    max_top1_regression: float = 0.0,
    seed_reports: Mapping[int, MovementEvaluationReportV1] | None = None,
    required_seeds: Sequence[int] = FORMAL_MOVEMENT_GATE_SEEDS,
    max_distribution_js: float = 0.10,
    max_mirror_error: float = 0.05,
) -> MovementGateReportV1:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if max_distribution_js < 0.0 or max_mirror_error < 0.0:
        raise ValueError("movement gate thresholds must be non-negative")
    requested_seeds = tuple(int(seed) for seed in required_seeds)
    if len(set(requested_seeds)) != len(requested_seeds) or not requested_seeds:
        raise ValueError("required_seeds must be non-empty and unique")
    normalized_seeds = FORMAL_MOVEMENT_GATE_SEEDS
    indeterminate = list(candidate.indeterminate)
    failures = list(candidate.failures)
    if requested_seeds != FORMAL_MOVEMENT_GATE_SEEDS:
        indeterminate.append(
            f"formal movement gate requires exactly seeds {list(FORMAL_MOVEMENT_GATE_SEEDS)} "
            f"(requested {list(requested_seeds)})"
        )
    for horizon_name, count in candidate.raw_counts.items():
        if count < minimum_count:
            indeterminate.append(f"{horizon_name}: raw count {count} is below minimum {minimum_count}")
    if baseline is not None:
        indeterminate.append(
            "separate train-prior and architecture baselines are required; legacy baseline is not accepted"
        )
    if train_prior_baseline is None:
        indeterminate.append("train-prior baseline is missing")
    if architecture_baseline is None:
        indeterminate.append("v2 A architecture baseline is missing")
    if architecture_baseline is not None:
        for horizon_name in _HORIZON_NAMES:
            candidate_top1, candidate_problem = _gate_metric_value(
                candidate.metrics.get(horizon_name, {}).get("move_top1", _MISSING)
            )
            baseline_top1, baseline_problem = _gate_metric_value(
                architecture_baseline.metrics.get(horizon_name, {}).get("move_top1", _MISSING)
            )
            if candidate_problem is not None:
                failures.append(f"{horizon_name}: candidate move_top1 is {candidate_problem}")
            if baseline_problem is not None:
                indeterminate.append(f"{horizon_name}: v2 A baseline move_top1 is {baseline_problem}")
            if candidate_top1 is not None and baseline_top1 is not None and candidate_top1 < baseline_top1 - max_top1_regression:
                failures.append(
                    f"{horizon_name}: move_top1 regressed from {baseline_top1:.6g} to {candidate_top1:.6g}"
                )

    seed_payload: dict[str, Any] = {}
    normalized_seed_reports: dict[int, MovementEvaluationReportV1] = {}
    if seed_reports is None:
        indeterminate.append(f"required seed reports are missing: {','.join(str(seed) for seed in normalized_seeds)}")
    else:
        normalized_seed_reports = {}
        for seed, report in seed_reports.items():
            if report is not None:
                normalized_seed_reports[int(seed)] = report
        missing_seeds = [seed for seed in normalized_seeds if seed not in normalized_seed_reports]
        extra_seeds = sorted(set(normalized_seed_reports) - set(normalized_seeds))
        if missing_seeds or extra_seeds:
            indeterminate.append(
                f"seed reports must exactly cover {list(normalized_seeds)} "
                f"(missing={missing_seeds}, extra={extra_seeds})"
            )
        for seed in normalized_seeds:
            report = normalized_seed_reports.get(seed)
            if report is None:
                continue
            indeterminate.extend(f"seed {seed}: {item}" for item in report.indeterminate)
            failures.extend(f"seed {seed}: {item}" for item in report.failures)
            seed_failures: list[str] = []
            for horizon_name, count in report.raw_counts.items():
                if count < minimum_count:
                    indeterminate.append(f"seed {seed} {horizon_name}: raw count {count} is below minimum {minimum_count}")
            for horizon_name in _HORIZON_NAMES:
                metrics = report.metrics.get(horizon_name, {})
                js_value = metrics.get("distribution_js")
                mirror_value = metrics.get("mirror_equivariance_error")
                class_count = metrics.get("predicted_move_class_count")
                js_numeric, js_problem = _gate_metric_value(js_value)
                if js_problem == "missing":
                    seed_failures.append(f"seed {seed} {horizon_name}: distribution JS exceeds {max_distribution_js:.6g}")
                elif js_problem is not None:
                    seed_failures.append(f"seed {seed} {horizon_name}: distribution_js is {js_problem}")
                elif js_numeric is not None and js_numeric > max_distribution_js:
                    seed_failures.append(f"seed {seed} {horizon_name}: distribution JS exceeds {max_distribution_js:.6g}")
                mirror_numeric, mirror_problem = _gate_metric_value(mirror_value)
                if mirror_problem == "missing":
                    seed_failures.append(f"seed {seed} {horizon_name}: mirror error exceeds {max_mirror_error:.6g}")
                elif mirror_problem is not None:
                    seed_failures.append(f"seed {seed} {horizon_name}: mirror_equivariance_error is {mirror_problem}")
                elif mirror_numeric is not None and mirror_numeric > max_mirror_error:
                    seed_failures.append(f"seed {seed} {horizon_name}: mirror error exceeds {max_mirror_error:.6g}")
                class_numeric, class_problem = _gate_metric_value(class_count)
                if class_count is None or class_numeric is None and class_problem == "missing":
                    seed_failures.append(f"seed {seed} {horizon_name}: move prediction collapsed to one class")
                elif class_problem is not None:
                    seed_failures.append(f"seed {seed} {horizon_name}: predicted_move_class_count is {class_problem}")
                elif class_numeric <= 1.0:
                    seed_failures.append(f"seed {seed} {horizon_name}: move prediction collapsed to one class")
            if seed_failures:
                failures.extend(seed_failures)
            seed_payload[str(seed)] = report.to_dict()

    for horizon_name in _HORIZON_NAMES:
        metrics = candidate.metrics.get(horizon_name, {})
        js_value = metrics.get("distribution_js")
        mirror_value = metrics.get("mirror_equivariance_error")
        js_numeric, js_problem = _gate_metric_value(js_value)
        if js_problem == "missing":
            failures.append(f"{horizon_name}: distribution JS exceeds {max_distribution_js:.6g}")
        elif js_problem is not None:
            failures.append(f"{horizon_name}: distribution_js is {js_problem}")
        elif js_numeric is not None and js_numeric > max_distribution_js:
            failures.append(f"{horizon_name}: distribution JS exceeds {max_distribution_js:.6g}")
        mirror_numeric, mirror_problem = _gate_metric_value(mirror_value)
        if mirror_problem == "missing":
            indeterminate.append(f"{horizon_name}: mirror-equivalence evidence is missing")
        elif mirror_problem is not None:
            failures.append(f"{horizon_name}: mirror_equivariance_error is {mirror_problem}")
        elif mirror_numeric is not None and mirror_numeric > max_mirror_error:
            failures.append(f"{horizon_name}: mirror error exceeds {max_mirror_error:.6g}")
        class_count = metrics.get("predicted_move_class_count")
        class_numeric, class_problem = _gate_metric_value(class_count)
        if class_count is None or class_numeric is None and class_problem == "missing":
            failures.append(f"{horizon_name}: move prediction collapsed to one class")
        elif class_problem is not None:
            failures.append(f"{horizon_name}: predicted_move_class_count is {class_problem}")
        elif class_numeric <= 1.0:
            failures.append(f"{horizon_name}: move prediction collapsed to one class")

    for name, label in (("stance_positive_count", "stance"), ("jump_positive_count", "jump")):
        positive_count, positive_problem = _gate_metric_value(candidate.metrics.get("horizon_0", {}).get(name, _MISSING))
        if positive_problem is not None and positive_problem != "missing":
            failures.append(f"horizon_0: {name} is {positive_problem}")
        elif positive_count is None or positive_count <= 0:
            indeterminate.append(f"horizon_0: no positive {label} samples")

    median_metrics: dict[str, dict[str, float]] = {}
    if normalized_seed_reports and all(seed in normalized_seed_reports for seed in normalized_seeds):
        def median_metric(horizon_name: str, metric_name: str) -> float | None:
            values: list[float] = []
            for seed in normalized_seeds:
                value, problem = _gate_metric_value(
                    normalized_seed_reports[seed].metrics.get(horizon_name, {}).get(metric_name, _MISSING)
                )
                if problem is not None or value is None:
                    return None
                values.append(value)
            return sorted(values)[len(values) // 2]

        if train_prior_baseline is not None:
            for horizon_index, horizon_name in enumerate(_HORIZON_NAMES):
                candidate_nll = median_metric(horizon_name, "move_nll")
                baseline_nll, baseline_problem = _gate_metric_value(
                    train_prior_baseline.metrics.get(horizon_name, {}).get("move_nll", _MISSING)
                )
                improvement = 0.10 if horizon_index == 0 else 0.05
                if candidate_nll is None or baseline_problem is not None or baseline_nll is None or baseline_nll <= 0.0:
                    indeterminate.append(f"{horizon_name}: NLL train-prior baseline or three-seed evidence is missing")
                else:
                    median_metrics.setdefault(horizon_name, {})["move_nll"] = candidate_nll
                    if candidate_nll > baseline_nll * (1.0 - improvement):
                        failures.append(
                            f"{horizon_name}: median move NLL {candidate_nll:.6g} did not improve "
                            f"by {improvement:.0%} over train-prior baseline {baseline_nll:.6g}"
                        )
                for seed in normalized_seeds:
                    seed_nll, seed_problem = _gate_metric_value(
                        normalized_seed_reports[seed].metrics.get(horizon_name, {}).get("move_nll", _MISSING)
                    )
                    if seed_problem is not None or seed_nll is None or baseline_nll is None:
                        indeterminate.append(f"seed {seed} {horizon_name}: NLL train-prior comparison is missing")
                    elif seed_nll >= baseline_nll:
                        failures.append(
                            f"seed {seed} {horizon_name}: move NLL {seed_nll:.6g} is not below "
                            f"train-prior baseline {baseline_nll:.6g}"
                        )

        if architecture_baseline is not None:
            for horizon_name, metric_name, label in (
                ("horizon_0", "move_circular_mae_deg", "direction circular MAE"),
                ("horizon_2", "displacement_error", "+250ms displacement error"),
            ):
                candidate_value = median_metric(horizon_name, metric_name)
                baseline_value, baseline_problem = _gate_metric_value(
                    architecture_baseline.metrics.get(horizon_name, {}).get(metric_name, _MISSING)
                )
                if candidate_value is None or baseline_problem is not None or baseline_value is None or baseline_value <= 0.0:
                    indeterminate.append(f"{horizon_name}: {label} v2 A baseline or three-seed evidence is missing")
                else:
                    median_metrics.setdefault(horizon_name, {})[metric_name] = candidate_value
                    if candidate_value > baseline_value * 0.90:
                        failures.append(
                            f"{horizon_name}: median {label} {candidate_value:.6g} did not improve "
                            f"by 10% over v2 A baseline {baseline_value:.6g}"
                        )

            for horizon_name, metric_name, label in (
                ("horizon_0", "stance_macro_f1", "stance macro-F1"),
                ("horizon_0", "jump_macro_f1", "jump macro-F1"),
            ):
                candidate_value = median_metric(horizon_name, metric_name)
                baseline_value, baseline_problem = _gate_metric_value(
                    architecture_baseline.metrics.get(horizon_name, {}).get(metric_name, _MISSING)
                )
                if candidate_value is None or baseline_problem is not None or baseline_value is None:
                    indeterminate.append(f"{horizon_name}: {label} v2 A baseline or three-seed evidence is missing")
                elif candidate_value < baseline_value:
                    failures.append(
                        f"{horizon_name}: median {label} {candidate_value:.6g} is below v2 A baseline {baseline_value:.6g}"
                    )
    return MovementGateReportV1(
        accepted=not indeterminate and not failures,
        raw_counts=candidate.raw_counts,
        confidence_intervals=candidate.confidence_intervals,
        indeterminate=tuple(dict.fromkeys(indeterminate)),
        failures=tuple(dict.fromkeys(failures)),
        evidence={
            "metrics": candidate.metrics,
            "worst_slice": candidate.worst_slice,
            "baseline_present": train_prior_baseline is not None and architecture_baseline is not None,
            "train_prior_baseline_present": train_prior_baseline is not None,
            "architecture_baseline_present": architecture_baseline is not None,
            "legacy_baseline_rejected": baseline is not None,
            "required_seeds": list(normalized_seeds),
            "seed_reports": seed_payload,
            "median_seed_metrics": median_metrics,
            "thresholds": {
                "max_distribution_js": max_distribution_js,
                "max_mirror_error": max_mirror_error,
                "nll_improvement": {"horizon_0": 0.10, "horizon_1": 0.05, "horizon_2": 0.05},
            },
            "single_aggregate_forbidden": True,
        },
    )


class SingleReadTestSplitV1:
    """A test split reader that makes tuning/test leakage fail closed."""

    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        self._records = tuple(records)
        self.read_count = 0

    def read_test(self, *, phase: str = "final") -> tuple[Mapping[str, Any], ...]:
        if phase != "final":
            raise RuntimeError("test split is locked during tuning; only final evaluation may read it")
        if self.read_count:
            raise RuntimeError("test split may be read only once")
        self.read_count += 1
        return self._records

    @property
    def test_read_count(self) -> int:
        return self.read_count


TestSplitReadAuditV1 = SingleReadTestSplitV1


def _manifest_value(manifest: Any, name: str, default: Any = None) -> Any:
    if isinstance(manifest, Mapping):
        return manifest.get(name, default)
    return getattr(manifest, name, default)


def _manifest_sha256(manifest: Any) -> str:
    for name in ("sha256", "source_sha256", "manifest_sha256"):
        value = _manifest_value(manifest, name)
        if value:
            return str(value)
    payload = json.dumps(_plain(manifest), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_movement_ablation_manifest(
    split_manifest: Any,
    *,
    seeds: Sequence[int] = FORMAL_MOVEMENT_GATE_SEEDS,
) -> dict[str, Any]:
    purpose = _manifest_value(split_manifest, "purpose")
    if hasattr(purpose, "value"):
        purpose = purpose.value
    if purpose is not None and str(purpose) != "test_only":
        raise ValueError("movement ablation manifests must be test_only")
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("at least one ablation seed is required")
    split_sha = _manifest_sha256(split_manifest)
    descriptions = {
        "A": "v2 history/memory/IntentV1 with Beta movement",
        "B": "current 256 features without history or memory with factorized single-horizon classification",
        "C": "legal entity tokens with factorized single-horizon classification",
        "D": "legal entity tokens with factorized three-horizon classification",
        "E": "D plus train-only 20-unit legal-position noise",
    }
    return {
        "schema": MOVEMENT_ABLATION_SCHEMA,
        "purpose": "test_only",
        "split_manifest_sha256": split_sha,
        "seeds": normalized_seeds,
        "experiments": {
            experiment_id: {
                "id": experiment_id,
                "description": descriptions[experiment_id],
                "split_manifest_sha256": split_sha,
                "seeds": list(normalized_seeds),
            }
            for experiment_id in ABLATION_IDS
        },
    }


build_ablation_manifest = build_movement_ablation_manifest


def _coerce_movement_report(value: Any) -> MovementEvaluationReportV1 | None:
    if value is None:
        return None
    if isinstance(value, MovementEvaluationReportV1):
        return value
    if isinstance(value, Mapping) and "metrics" in value:
        return MovementEvaluationReportV1(
            metrics=value.get("metrics", {}),
            raw_counts=value.get("raw_counts", {}),
            confidence_intervals=value.get("confidence_intervals", {}),
            slices=value.get("slices", {}),
            worst_slice=value.get("worst_slice", {}),
            indeterminate=tuple(value.get("indeterminate", ())),
            failures=tuple(value.get("failures", ())),
        )
    if isinstance(value, Mapping):
        return evaluate_movement_predictions((value,))
    return evaluate_movement_predictions(value)


def evaluate_movement_ablation(
    experiment_records: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    split_manifest: Any,
    seeds: Sequence[int] = (7, 17, 29),
    test_split: SingleReadTestSplitV1 | None = None,
    phase: str = "final",
    train_prior_baseline: Any = None,
    architecture_baseline: Any = None,
    seed_reports: Mapping[int, Any] | None = None,
    gate_experiment: str = "D",
) -> dict[str, Any]:
    manifest = build_movement_ablation_manifest(split_manifest, seeds=seeds)
    unknown = sorted(set(experiment_records) - set(ABLATION_IDS))
    missing = sorted(set(ABLATION_IDS) - set(experiment_records))
    if unknown or missing:
        raise ValueError(f"ablation experiments must be exactly A-E (missing={missing}, unknown={unknown})")
    normalized_records = {experiment_id: tuple(rows) for experiment_id, rows in experiment_records.items()}
    if test_split is not None:
        test_rows = test_split.read_test(phase=phase)
        if not any(normalized_records.values()):
            normalized_records = {experiment_id: test_rows for experiment_id in ABLATION_IDS}
    report_objects = {
        experiment_id: evaluate_movement_predictions(rows)
        for experiment_id, rows in normalized_records.items()
    }
    if gate_experiment not in report_objects:
        raise ValueError(f"gate experiment must be one of {ABLATION_IDS}")
    normalized_seed_reports = None
    if seed_reports is not None:
        normalized_seed_reports = {
            int(seed): _coerce_movement_report(report)
            for seed, report in seed_reports.items()
        }
    gate = build_movement_gate_report(
        report_objects[gate_experiment],
        train_prior_baseline=_coerce_movement_report(train_prior_baseline),
        architecture_baseline=_coerce_movement_report(architecture_baseline),
        seed_reports=normalized_seed_reports,
        required_seeds=FORMAL_MOVEMENT_GATE_SEEDS,
    )
    return {
        "schema": MOVEMENT_ABLATION_SCHEMA,
        "manifest": manifest,
        "experiments": {experiment_id: report.to_dict() for experiment_id, report in report_objects.items()},
        "gate": gate.to_dict(),
        "aggregate": None,
        "aggregate_note": "Per-experiment and worst-slice evidence is authoritative; no single aggregate is emitted.",
    }


def movement_evidence_is_accepted(evidence: Any) -> bool | None:
    """Return only a validated strict movement gate state."""

    if evidence is None:
        return None
    if not isinstance(evidence, Mapping):
        return False
    gate = evidence.get("gate", evidence)
    if not isinstance(gate, Mapping) or gate.get("schema") != "movement-gate-v1":
        return False
    accepted = gate.get("accepted")
    if not isinstance(accepted, bool):
        return False
    failures = gate.get("failures")
    indeterminate = gate.get("indeterminate")
    strict_evidence = gate.get("evidence")
    if not isinstance(failures, (list, tuple)) or not isinstance(indeterminate, (list, tuple)):
        return False
    if not isinstance(strict_evidence, Mapping):
        return False
    if not accepted:
        return False
    required_seeds = strict_evidence.get("required_seeds")
    seed_reports = strict_evidence.get("seed_reports")
    if strict_evidence.get("single_aggregate_forbidden") is not True:
        return False
    if strict_evidence.get("train_prior_baseline_present") is not True:
        return False
    if strict_evidence.get("architecture_baseline_present") is not True:
        return False
    if not isinstance(required_seeds, list) or not required_seeds:
        return False
    if not isinstance(seed_reports, Mapping):
        return False
    try:
        normalized_required_seeds = tuple(int(seed) for seed in required_seeds)
    except (TypeError, ValueError):
        return False
    if normalized_required_seeds != FORMAL_MOVEMENT_GATE_SEEDS:
        return False
    if set(str(seed) for seed in FORMAL_MOVEMENT_GATE_SEEDS) != set(seed_reports):
        return False
    return not failures and not indeterminate


__all__ = [
    "ABLATION_IDS",
    "FORMAL_MOVEMENT_GATE_SEEDS",
    "MOVEMENT_ABLATION_SCHEMA",
    "MOVEMENT_EVALUATION_SCHEMA",
    "MOVEMENT_MIRROR_PERMUTATION",
    "MovementEvaluationReportV1",
    "MovementGateReportV1",
    "SingleReadTestSplitV1",
    "TestSplitReadAuditV1",
    "adapt_beta_movement_to_labels",
    "adapt_v2_beta_to_movement_labels",
    "build_ablation_manifest",
    "build_movement_ablation_manifest",
    "build_movement_gate_report",
    "brier_score",
    "circular_direction_error",
    "displacement_error",
    "ece",
    "evaluate_movement_ablation",
    "evaluate_movement_predictions",
    "expected_calibration_error",
    "js_divergence",
    "macro_f1",
    "mean_displacement_error",
    "mirror_equivariance_error",
    "mirror_error",
    "movement_evidence_is_accepted",
    "unified_movement_labels_from_v2",
]

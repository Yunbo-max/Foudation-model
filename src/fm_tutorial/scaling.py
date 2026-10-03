"""Small, identifiable single-axis fits to user-supplied measured CSV data.

The fit log(loss) = intercept + exponent*log(axis) has no irreducible-loss
parameter. It is a local descriptive model, not a Chinchilla optimum estimator.
"""
import csv
import math
from pathlib import Path

import numpy as np


def read_measurements(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"params", "tokens", "loss"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for line_number, row in enumerate(reader, 2):
            try:
                measurement = {key: float(row[key]) for key in required}
            except (ValueError, TypeError) as error:
                raise ValueError(f"CSV line {line_number}: params, tokens, loss must be numbers") from error
            if "split" in row:
                measurement["split"] = (row["split"] or "").strip().lower()
            rows.append(measurement)
    return rows


def fit_power_law(records, axis="tokens"):
    """Fit train rows and evaluate untouched held-out measurements.

    Required: positive finite params/tokens/loss; the other axis fixed; at least
    three distinct train x values and a held-out row. Explicit split values are
    ``train`` and ``holdout``. Without them, reserve the largest fifth of rows
    by axis as an extrapolation holdout (at least one row).
    """
    if axis not in ("params", "tokens"):
        raise ValueError("axis must be params or tokens")
    records = list(records)
    if len(records) < 4:
        raise ValueError("Need at least four measured rows (three training and one held-out)")
    rows = []
    for index, record in enumerate(records):
        try:
            row = {key: float(record[key]) for key in ("params", "tokens", "loss")}
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"Row {index + 1} requires numeric params, tokens, loss") from error
        if any(not math.isfinite(value) or value <= 0 for value in row.values()):
            raise ValueError(f"Row {index + 1} requires positive finite params, tokens, loss")
        row["split"] = record.get("split", "")
        rows.append(row)
    controlled = "params" if axis == "tokens" else "tokens"
    fixed = rows[0][controlled]
    if any(not math.isclose(row[controlled], fixed, rel_tol=1e-9) for row in rows):
        raise ValueError(f"Single-axis fit requires fixed {controlled}; select one controlled sweep instead of fitting confounded rows")
    if any(row["split"] for row in rows):
        if any(row["split"] not in ("train", "holdout") for row in rows):
            raise ValueError("Every split must be train or holdout when split labels are supplied")
        train = [row for row in rows if row["split"] == "train"]
        holdout = [row for row in rows if row["split"] == "holdout"]
        split_method = "explicit CSV split"
    else:
        rows.sort(key=lambda row: row[axis])
        held_count = max(1, math.ceil(len(rows) * 0.2))
        train, holdout = rows[:-held_count], rows[-held_count:]
        split_method = "largest-axis 20% extrapolation holdout"
    if len({row[axis] for row in train}) < 3 or not holdout:
        raise ValueError("Need three distinct training axis values and a nonempty held-out partition")
    log_x = np.log([row[axis] for row in train])
    log_y = np.log([row["loss"] for row in train])
    centered_x = log_x - log_x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 1e-12:
        raise ValueError("Training axis values have insufficient log-space variation")
    exponent = float(np.dot(centered_x, log_y - log_y.mean()) / denominator)
    intercept = float(log_y.mean() - exponent * log_x.mean())
    evaluations = []
    errors = []
    for row in holdout:
        log_prediction = intercept + exponent * math.log(row[axis])
        try:
            prediction = math.exp(log_prediction)
        except OverflowError as error:
            raise ValueError("Fit extrapolation overflows; narrow the sweep or inspect measurements") from error
        errors.append(log_prediction - math.log(row["loss"]))
        evaluations.append({"params": row["params"], "tokens": row["tokens"], "loss": row["loss"], "prediction": prediction})
    try:
        coefficient = math.exp(intercept)
    except OverflowError as error:
        raise ValueError("Fitted coefficient overflows; rescale units or narrow the sweep") from error
    return {"model": "loss = coefficient * axis ** exponent", "axis": axis, "controlled_axis": controlled,
            "controlled_value": fixed, "coefficient": coefficient, "exponent": exponent,
            "train_count": len(train), "holdout_count": len(holdout), "split_method": split_method,
            "train_axis_range": [min(row[axis] for row in train), max(row[axis] for row in train)],
            "train_log_rmse": float(np.sqrt(np.mean((intercept + exponent * log_x - log_y) ** 2))),
            "holdout_log_rmse": float(np.sqrt(np.mean(np.square(errors)))), "holdout": evaluations,
            "limitations": "Local single-axis description; no irreducible-loss estimate, uncertainty interval, compute-optimal allocation, or causal claim. CSV provenance must be supplied by the user."}

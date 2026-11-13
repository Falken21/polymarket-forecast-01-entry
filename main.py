"""Adaptive quantile forecaster for Polymarket's "Next CEO of NVIDIA?" event.

For every case file in the input directory, forecast each market's Yes price
for the 240 minutes after the checkpoint as five quantiles (0.05, 0.25, 0.5,
0.75, 0.95). Unlike the fixed house baseline, each market's spread is
estimated from that market's own one-minute moves inside the case: quiet
markets get tight quantiles, jumpy markets get wide ones. The median stays
at the checkpoint price — on the practice cases a drift term only chased
jumps and measurably worsened the pinball loss, so none is used.
Standard library only; fully deterministic.
"""

import json
import math
import os
from pathlib import Path

Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
INNER_WIDEN = 1.05  # 25% / 75% quantiles sit slightly wider than Gaussian
SIGMA_FLOOR = 0.0012
SIGMA_CAP = 0.012
RECENT_MIN = 360


def mean_abs(diffs):
    return sum(abs(d) for d in diffs) / len(diffs) if diffs else 0.0


def estimate_sigma(prices):
    """Per-market step size: blend of the recent window and the full case."""
    diffs = [b - a for a, b in zip(prices, prices[1:])]
    if not diffs:
        return 0.0025
    to_sigma = math.sqrt(math.pi / 2.0)  # E|X| -> sigma for a normal step
    sigma_full = mean_abs(diffs) * to_sigma
    sigma_recent = mean_abs(diffs[-RECENT_MIN:]) * to_sigma
    sigma = 0.65 * sigma_recent + 0.35 * sigma_full
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_market(prices, horizon):
    x = prices[-1]
    sigma = estimate_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        center = x
        s = sigma * math.sqrt(h)
        raw = [
            center + Z[0] * s,
            center + Z[1] * INNER_WIDEN * s,
            center,
            center + Z[3] * INNER_WIDEN * s,
            center + Z[4] * s,
        ]
        row = [round(min(1.0, max(0.0, v)), 6) for v in raw]
        for i in range(1, 5):  # quantiles must be non-decreasing
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def run_case(case):
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = [
        {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
        for m in case["markets"]
    ]
    return {"forecasts": forecasts}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = run_case(case)
        out_path = out_dir / (path.stem + ".out")
        out_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()

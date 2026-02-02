import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def _load_labels(prompt_config_path: Path) -> List[str]:
    cfg = OmegaConf.load(prompt_config_path)
    # Support both formats: top-level `labels` or nested `prompt.labels`
    labels = list(cfg.get("labels", []))
    if not labels:
        labels = list(cfg.get("prompt", {}).get("labels", []))
    if not labels:
        raise ValueError(f"No labels found under prompt.labels in {prompt_config_path}")
    return labels


def _allocate_samples(
    counts: Dict[Tuple[int, ...], int],
    target_n: int,
) -> Dict[Tuple[int, ...], int]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}

    # Initial proportional allocation
    raw = {k: (target_n * v / total) for k, v in counts.items()}
    alloc = {k: int(raw[k]) for k in counts}

    # Distribute remaining by largest fractional parts, respecting group capacity
    remainder = target_n - sum(alloc.values())
    if remainder > 0:
        fractional = sorted(
            ((k, raw[k] - alloc[k]) for k in counts),
            key=lambda x: x[1],
            reverse=True,
        )
        idx = 0
        while remainder > 0 and fractional:
            k, _ = fractional[idx % len(fractional)]
            if alloc[k] < counts[k]:
                alloc[k] += 1
                remainder -= 1
            idx += 1
            # If we've looped too much without progress, break
            if idx > len(fractional) * 5 and remainder > 0:
                break

    return alloc


def stratified_multilabel_sample(
    df: pd.DataFrame,
    label_cols: List[str],
    n: int,
    seed: int,
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be > 0")
    if n > len(df):
        log.warning("Requested n > dataset size; returning full dataset.")
        return df.copy()

    # Build multilabel combo key
    label_matrix = df[label_cols].fillna(0).astype(int)
    combos = list(map(tuple, label_matrix.values.tolist()))
    df = df.copy()
    df["__combo__"] = combos

    counts = df["__combo__"].value_counts().to_dict()
    alloc = _allocate_samples(counts, n)

    parts = []
    for combo, k in alloc.items():
        if k <= 0:
            continue
        group = df[df["__combo__"] == combo]
        sample = group.sample(n=min(k, len(group)), random_state=seed)
        parts.append(sample)

    sampled = pd.concat(parts, ignore_index=True) if parts else df.head(0)

    # If rounding/availability left us short, top up randomly from remaining rows
    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        if not remaining.empty:
            top_up = remaining.sample(
                n=min(n - len(sampled), len(remaining)),
                random_state=seed,
            )
            sampled = pd.concat([sampled, top_up], ignore_index=True)

    sampled = sampled.drop(columns=["__combo__"]).reset_index(drop=True)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified tiny_tuning_set from tuning_set.csv using prompt.labels",
    )
    parser.add_argument(
        "--input",
        default="data/tuning_set.csv",
        help="Input CSV path (default: data/tuning_set.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/tiny_tuning_set.csv",
        help="Output CSV path (default: data/tiny_tuning_set.csv)",
    )
    parser.add_argument(
        "--prompt",
        default="configs/prompt/3-shot_multi_v2.yaml",
        help="Prompt config path containing prompt.labels",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of rows to sample (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    prompt_path = Path(args.prompt)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt config not found: {prompt_path}")

    labels = _load_labels(prompt_path)

    df = pd.read_csv(input_path)
    missing = [c for c in labels if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing label columns in CSV: {missing}")

    sampled = stratified_multilabel_sample(df, labels, args.n, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)

    log.info("Wrote %d rows to %s", len(sampled), output_path)

    # Report label prevalence in input vs output
    def _prevalence(table: pd.DataFrame) -> Dict[str, float]:
        rates = {}
        for label in labels:
            rates[label] = float(table[label].fillna(0).astype(int).mean() * 100.0)
        return rates

    input_prev = _prevalence(df)
    output_prev = _prevalence(sampled)

    print("\nLabel prevalence (%):")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    for label in labels:
        print(f"- {label}: input {input_prev[label]:.2f}% | output {output_prev[label]:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()

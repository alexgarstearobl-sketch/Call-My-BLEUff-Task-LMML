"""Analyze Call My BLEUff decks and decode the selected seats.

Run from any directory:
    python call_my_bleuff.py

The selection is a heuristic, not proof of intent: lexical overlap with the
references and repetitive/overlong output are observable clues, but the CSVs
do not contain ground-truth labels for honest panelists.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from bleu import corpus_bleu, segment_stats, tokenize


HERE = Path(__file__).resolve().parent
EXPECTED_ROUNDS = set(range(1, 25))
EXPECTED_CARDS = set(range(1, 41))
EXPECTED_SEATS = {1, 2, 3}


def load_data(references_path: Path, panel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    references = pd.read_csv(references_path)
    panel = pd.read_csv(panel_path)

    ref_columns = ["round", "card", "reference"]
    panel_columns = ["round", "card", "seat", "text"]
    if not set(ref_columns).issubset(references.columns):
        raise ValueError(f"{references_path} must contain columns {ref_columns}")
    if not set(panel_columns).issubset(panel.columns):
        raise ValueError(f"{panel_path} must contain columns {panel_columns}")

    references = references[ref_columns].copy()
    panel = panel[panel_columns].copy()
    if references.isna().any().any() or panel.isna().any().any():
        raise ValueError("Input CSVs contain empty required fields.")

    for frame, keys, label in (
        (references, ["round", "card", "reference"], "references"),
        (panel, ["round", "card", "seat"], "panel"),
    ):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label}.csv contains duplicate records for {keys}.")

    references["round"] = references["round"].astype(int)
    references["card"] = references["card"].astype(int)
    panel["round"] = panel["round"].astype(int)
    panel["card"] = panel["card"].astype(int)
    panel["seat"] = panel["seat"].astype(int)

    if set(references["round"]) != EXPECTED_ROUNDS or set(panel["round"]) != EXPECTED_ROUNDS:
        raise ValueError("Both files must contain rounds 1 through 24.")
    if set(references["card"]) != EXPECTED_CARDS or set(panel["card"]) != EXPECTED_CARDS:
        raise ValueError("Both files must contain cards 1 through 40.")
    if set(panel["seat"]) != EXPECTED_SEATS:
        raise ValueError("panel.csv must contain seats 1, 2, and 3.")

    ref_counts = references.groupby(["round", "card"]).size()
    if len(ref_counts) != 24 * 40 or not ref_counts.eq(5).all():
        raise ValueError("Every round/card pair must have exactly five references.")
    seat_sets = panel.groupby(["round", "card"])["seat"].agg(lambda values: set(values))
    if len(seat_sets) != 24 * 40 or not seat_sets.map(lambda seats: seats == EXPECTED_SEATS).all():
        raise ValueError("Every round/card pair must have exactly one row for each seat.")

    return references, panel


def deck_statistics(
    round_number: int,
    seat: int,
    deck: pd.DataFrame,
    reference_map: dict[tuple[int, int], list[str]],
) -> dict[str, Any]:
    deck = deck.sort_values("card")
    outputs = deck["text"].astype(str).tolist()
    references = [reference_map[(round_number, int(card))] for card in deck["card"]]
    bleu = corpus_bleu(outputs, references)
    stats = [segment_stats(output, refs) for output, refs in zip(outputs, references)]

    matches = [sum(item["matched"][order] for item in stats) for order in range(4)]
    totals = [sum(item["total"][order] for item in stats) for order in range(4)]
    precisions = [matched / total if total else 0.0 for matched, total in zip(matches, totals)]

    token_lists = [tokenize(text) for text in outputs]
    lengths = [len(tokens) for tokens in token_lists]
    token_count = sum(lengths)
    repeated = sum(
        1.0 - len(set(tokens)) / len(tokens)
        for tokens in token_lists
        if tokens
    ) / max(1, sum(bool(tokens) for tokens in token_lists))
    vocabulary = Counter(token for tokens in token_lists for token in tokens)
    vocab_size = len(vocabulary)
    entropy = (
        -sum((count / token_count) * math.log2(count / token_count) for count in vocabulary.values())
        if token_count
        else 0.0
    )

    return {
        "round": round_number,
        "seat": seat,
        "bleu": bleu["bleu"],
        "p1": precisions[0],
        "p2": precisions[1],
        "p3": precisions[2],
        "p4": precisions[3],
        "mean_tokens": sum(lengths) / max(1, len(lengths)),
        "repetition": repeated,
        "vocab_entropy": entropy,
        "vocab_size": vocab_size,
    }


def add_bluff_scores(decks: pd.DataFrame) -> pd.DataFrame:
    """Score observable copying and text-stuffing signals within each round."""
    scored = decks.copy()
    # Higher-order overlap is more specific to copied phrases than unigram overlap.
    scored["reference_copy"] = 0.4 * scored["p3"] + 0.6 * scored["p4"]

    # Only excess repetition/length versus the other two seats is penalized.
    # This avoids treating an entire round's naturally longer descriptions as
    # suspicious simply because they are longer than other rounds.
    scored["bluff_score"] = 0.0
    for _, indices in scored.groupby("round").groups.items():
        group = scored.loc[indices]
        median_repetition = group["repetition"].median()
        median_length = group["mean_tokens"].median()
        length_mad = (group["mean_tokens"] - median_length).abs().median()
        length_scale = max(float(length_mad), 1.0)
        scored.loc[indices, "bluff_score"] = (
            group["reference_copy"]
            + 0.5 * (group["repetition"] - median_repetition).clip(lower=0)
            + 0.05 * ((group["mean_tokens"] - median_length) / length_scale).clip(lower=0)
        )
    return scored


def decode_seats(seats: list[int]) -> str:
    if len(seats) != 24 or any(seat not in EXPECTED_SEATS for seat in seats):
        raise ValueError("Decoding requires exactly 24 seat numbers, each in {1, 2, 3}.")

    letters = []
    for index in range(0, 24, 3):
        d1, d2, d3 = (seat - 1 for seat in seats[index : index + 3])
        value = 9 * d1 + 3 * d2 + d3
        if value > 25:
            raise ValueError(
                f"Rounds {index + 1}-{index + 3} decode to {value}, outside A-Z; "
                "the selected seats do not yield a valid flag word."
            )
        letters.append(chr(ord("A") + value))
    return "".join(letters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=HERE / "references.csv")
    parser.add_argument("--panel", type=Path, default=HERE / "panel.csv")
    args = parser.parse_args()

    references, panel = load_data(args.references, args.panel)
    reference_map = (
        references.sort_values(["round", "card"])
        .groupby(["round", "card"])["reference"]
        .apply(list)
        .to_dict()
    )

    records = [
        deck_statistics(round_number, seat, deck, reference_map)
        for (round_number, seat), deck in panel.groupby(["round", "seat"], sort=True)
    ]
    decks = add_bluff_scores(pd.DataFrame(records))

    print("Official corpus BLEU by round and seat:")
    print(decks.pivot(index="round", columns="seat", values="bleu").round(3).to_string())
    print("\nAlternative indicators and heuristic bluff score:")
    columns = [
        "round", "seat", "bleu", "reference_copy", "repetition",
        "mean_tokens", "vocab_entropy", "bluff_score",
    ]
    print(decks[columns].sort_values(["round", "seat"]).round(3).to_string(index=False))

    selected = (
        decks.sort_values(["round", "bluff_score", "seat"])
        .groupby("round", sort=True)
        .first()
        .reset_index()
    )
    seats = selected["seat"].astype(int).tolist()
    print("\nSelected seat by round (lowest heuristic bluff score):")
    print(" ".join(f"{round_number}:{seat}" for round_number, seat in zip(range(1, 25), seats)))
    try:
        print("Heuristic flag candidate (unconfirmed): SIGMOID_" + decode_seats(seats))
    except ValueError as error:
        print(f"Flag candidate unavailable: {error}")
    print(
        "\nNote: this is an unsupervised heuristic. Similarity and style can flag "
        "suspicious text, but cannot prove honesty or factuality without labels "
        "or independent venue facts."
    )


if __name__ == "__main__":
    main()

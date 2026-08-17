"""Compares two completed reviewer packets (see reviewer_packets.py),
reports agreement rate and per-case disagreements, and -- only for cases
where both reviewers agree -- writes a reconciled answer key.

This does NOT overwrite cases.json's constructed `label` automatically:
that's a deliberate manual step (see the note printed at the end), because
splicing reviewer labels back into the frozen dataset is exactly the kind
of change that should be visible in a diff and a commit message, not a side
effect of running a script.

Usage:
    python -m benchmarks.grounding.reconcile_reviews \\
        packets/sealed_review_alice.json packets/sealed_review_bob.json
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

RECONCILED_DIR = Path(__file__).parent / "packets"


def load_packet(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text())
    by_id = {c["id"]: c for c in data["cases"]}
    unlabeled = [cid for cid, c in by_id.items() if not c.get("reviewer_label")]
    if unlabeled:
        raise ValueError(
            f"{path} has {len(unlabeled)} case(s) with no reviewer_label set yet "
            f"(e.g. {unlabeled[0]}) -- finish labeling before reconciling."
        )
    return by_id, data["reviewer"], data["split"]


def reconcile(path_a: Path, path_b: Path) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    cases_a, reviewer_a, split_a = load_packet(path_a)
    cases_b, reviewer_b, split_b = load_packet(path_b)

    if split_a != split_b:
        raise ValueError(f"packets are for different splits: {split_a!r} vs {split_b!r}")
    if set(cases_a) != set(cases_b):
        raise ValueError(
            "packets don't cover the same case ids -- were they generated from the "
            "same cases.json? "
            f"only in {path_a.name}: {set(cases_a) - set(cases_b)}; "
            f"only in {path_b.name}: {set(cases_b) - set(cases_a)}"
        )

    agreed: List[Dict[str, Any]] = []
    disagreements: List[str] = []
    for case_id in sorted(cases_a):
        label_a = cases_a[case_id]["reviewer_label"]
        label_b = cases_b[case_id]["reviewer_label"]
        if label_a == label_b:
            agreed.append({"id": case_id, "label": label_a})
        else:
            disagreements.append(
                f"{case_id}: {reviewer_a}={label_a!r} ({cases_a[case_id]['reviewer_reasoning']!r}) "
                f"vs {reviewer_b}={label_b!r} ({cases_b[case_id]['reviewer_reasoning']!r})"
            )

    summary = {
        "split": split_a,
        "reviewers": [reviewer_a, reviewer_b],
        "case_count": len(cases_a),
        "agreed_count": len(agreed),
        "disagreement_count": len(disagreements),
        "agreement_rate": len(agreed) / len(cases_a) if cases_a else None,
    }
    return disagreements, agreed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_a", type=Path)
    parser.add_argument("packet_b", type=Path)
    args = parser.parse_args()

    disagreements, agreed, summary = reconcile(args.packet_a, args.packet_b)

    print(f"Split: {summary['split']}")
    print(f"Reviewers: {summary['reviewers'][0]!r}, {summary['reviewers'][1]!r}")
    print(f"Agreement rate: {summary['agreement_rate']:.1%} "
          f"({summary['agreed_count']}/{summary['case_count']})")

    gate = 0.90  # matches gates.py's human_verifier_agreement_rate gate
    if summary["agreement_rate"] is not None and summary["agreement_rate"] < gate:
        print(f"\nBelow the {gate:.0%} agreement gate -- per LABELING.md rule 4, discuss and "
              f"resolve every disagreement (or drop the case) before treating this split as sealed.")

    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s) to resolve:")
        for line in disagreements:
            print(f"  {line}")

    if agreed:
        out_path = RECONCILED_DIR / f"{summary['split']}_reconciled.json"
        out_path.write_text(json.dumps({
            "split": summary["split"],
            "reviewers": summary["reviewers"],
            "reconciled_labels": agreed,
        }, indent=2) + "\n")
        print(f"\nWrote {len(agreed)} agreed label(s) to {out_path}.")
        print(
            "This file is NOT applied to cases.json automatically. Once every "
            "disagreement above is resolved (by discussion, not a tie-break vote -- "
            "see LABELING.md rule 4), splice reconciled_labels into cases.json's "
            "`label` field as a reviewed, deliberate, and separately committed change, "
            "and only then rename DATASET_VERSION from '...-v1-candidate' to a plain "
            "'...-v1' in generate_dataset.py."
        )


if __name__ == "__main__":
    main()

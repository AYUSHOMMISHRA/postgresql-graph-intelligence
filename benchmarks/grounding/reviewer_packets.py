"""Produces a blinded worksheet for a human reviewer to label the `sealed`
split (or any split, via --split) without seeing the dataset's constructed
`label`, `supporting_quotes`, or `notes` -- exactly LABELING.md rule 4's
prerequisite: two *independent* reviewers, each blind to the answer key and
to each other's answers.

Usage:
    python -m benchmarks.grounding.reviewer_packets --reviewer alice
    python -m benchmarks.grounding.reviewer_packets --reviewer bob

Produces packets/sealed_review_<reviewer>.json: a copy of every sealed case
with the answer-key fields removed and two blank fields added for the
reviewer to fill in (`reviewer_label`, `reviewer_reasoning`). Once both
reviewers have filled in their copies, see reconcile_reviews.py.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

CASES_PATH = Path(__file__).parent / "cases.json"
PACKETS_DIR = Path(__file__).parent / "packets"

# Fields that constitute the answer key -- removed so the reviewer judges
# only from what a real system would have shown them (question, claim,
# citations, evidence), never from how the case was constructed.
ANSWER_KEY_FIELDS = {"label", "supporting_quotes", "notes"}


def load_cases(split: str) -> List[Dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text())
    return [c for c in data["cases"] if c["split"] == split]


def blind_case(case: Dict[str, Any]) -> Dict[str, Any]:
    blinded = {k: v for k, v in case.items() if k not in ANSWER_KEY_FIELDS}
    blinded["reviewer_label"] = None  # reviewer fills in: "supported" | "contradicted" | "insufficient"
    blinded["reviewer_reasoning"] = ""  # reviewer fills in: one sentence, required for disagreement review
    return blinded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True, help="Reviewer identifier, e.g. a name or initials.")
    parser.add_argument("--split", default="sealed", choices=["dev", "calibration", "sealed"])
    args = parser.parse_args()

    cases = load_cases(args.split)
    packet = {
        "split": args.split,
        "reviewer": args.reviewer,
        "instructions": (
            "For each case, read `question`, `claim`, `asserted_citation_ids`, and "
            "`retrieved_evidence` only -- do not search for outside information about "
            "whether the claim is true in reality. Set `reviewer_label` to one of "
            "'supported' | 'contradicted' | 'insufficient' per the definitions in "
            "LABELING.md, and fill in `reviewer_reasoning` with one sentence. Do not "
            "share your answers with the other reviewer until both packets are complete "
            "-- see reconcile_reviews.py for what happens next."
        ),
        "cases": [blind_case(c) for c in cases],
    }

    PACKETS_DIR.mkdir(exist_ok=True)
    out_path = PACKETS_DIR / f"{args.split}_review_{args.reviewer}.json"
    out_path.write_text(json.dumps(packet, indent=2) + "\n")
    print(f"Wrote blinded {args.split} packet ({len(cases)} cases) for reviewer "
          f"{args.reviewer!r} to {out_path}")


if __name__ == "__main__":
    main()

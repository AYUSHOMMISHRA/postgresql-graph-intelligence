"""Fixtures for a future verifier's exact-supporting-quote validation layer
(PR 3, item 5 in the Release 2 plan: "request quotes and locate them
server-side... reject quotes not found verbatim"). No verifier exists yet
to run these against, so this only builds and validates the fixture data
itself -- `fabricated_quote_rejection_rate` in gates.py stays N/A until a
verifier consumes these.

Each fixture pairs one real, literal-substring quote with one fabricated
quote that reads as plausible paraphrase but is *not* a literal substring
of the cited chunk -- exactly what a model asked to "quote the supporting
text" tends to produce when it paraphrases instead of quoting.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_VERSION = "grounding-fabrication-fixtures-v1-candidate"
OUTPUT_PATH = Path(__file__).parent / "fabrication_fixtures.json"


def _fixture(fixture_id: str, chunk_content: str, real_quote: str, fabricated_quote: str) -> Dict[str, Any]:
    assert real_quote in chunk_content, f"{fixture_id}: real_quote must be a literal substring of chunk_content"
    assert fabricated_quote not in chunk_content, (
        f"{fixture_id}: fabricated_quote must NOT be a literal substring of chunk_content "
        f"(that would make it a real quote, defeating the point of this fixture)"
    )
    return {
        "id": fixture_id,
        "chunk_content": chunk_content,
        "real_quote": real_quote,
        "fabricated_quote": fabricated_quote,
    }


def build_fixtures() -> List[Dict[str, Any]]:
    fixtures = []
    for n in range(1, 11):
        checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
        chunk = f"{checkout} depends on {auth} for session validation."
        fixtures.append(_fixture(
            f"paraphrase-not-quote-{n:03d}",
            chunk,
            real_quote=f"{checkout} depends on {auth}",
            # Plausible paraphrase a model might emit as "the quote" instead
            # of the literal text -- reordered/reworded, not a substring.
            fabricated_quote=f"{auth} is a dependency of {checkout}",
        ))
        fixtures.append(_fixture(
            f"fabricated-detail-not-in-chunk-{n:03d}",
            chunk,
            real_quote=f"{checkout} depends on {auth} for session validation",
            # Invents a specific detail (a percentage) the chunk never stated.
            fabricated_quote=f"{checkout} depends on {auth} for 99.9% of session validation",
        ))
    return fixtures


def main() -> None:
    fixtures = build_fixtures()
    OUTPUT_PATH.write_text(json.dumps(
        {"dataset_version": DATASET_VERSION, "fixture_count": len(fixtures), "fixtures": fixtures},
        indent=2,
    ) + "\n")
    print(f"Wrote {len(fixtures)} fixtures to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

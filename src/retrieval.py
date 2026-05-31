"""ColBERTv2 retrieval configuration for the multi-hop program.

Default: DSPy's hosted ColBERTv2 index over Wikipedia 2017 abstracts (the
standard HotpotQA retrieval substrate). Known to be flaky; the documented
fallback per the handoff is HotpotQA `distractor` mode, which provides gold
paragraphs inline and drops the create_query_hop2 module from the optimization
surface. The distractor fallback is not implemented here -- it is a Chunk-2
escape hatch only invoked if the hosted index is down for the whole build
window. See gepa_band_selection_handoff.md Section 9.

Passage long_text format (wiki17): `'"Title of Article" | First sentence...
Second sentence...'`. `parse_title` extracts the title segment.
"""

from __future__ import annotations

import dspy

HOSTED_COLBERTV2_URL = "http://20.102.90.50:2017/wiki17_abstracts"


def configure_retrieval(url: str = HOSTED_COLBERTV2_URL, k: int = 5) -> dspy.ColBERTv2:
    rm = dspy.ColBERTv2(url=url)
    dspy.settings.configure(rm=rm)
    return rm


def parse_title(passage: str) -> str:
    head, sep, _ = passage.partition(" | ")
    if not sep:
        return passage.strip().strip('"').strip()
    return head.strip().strip('"').strip()


def parse_titles(passages: list[str]) -> list[str]:
    return [parse_title(p) for p in passages]

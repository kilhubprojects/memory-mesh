"""Reciprocal Rank Fusion (RRF) for hybrid search.

RRF merges two ranked lists into a single fused ranking without requiring
score normalisation.  Given rank ``r`` (1-based) in a list, the contribution
is ``1 / (k + r)``.  Documents appearing in both lists accumulate scores from
both.

Reference: Cormack, Clarke, Buettcher (2009).
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using RRF.

    Args:
        rankings: Each inner list is a ranked sequence of document IDs
            (first element = rank 1, best).  IDs are arbitrary strings.
        k: RRF smoothing constant.  Higher k reduces the impact of top ranks.
            Cormack et al. recommend 60 for web search; works well for RAG.

    Returns:
        List of ``(doc_id, rrf_score)`` sorted by descending score.  The
        scale is ``1/(k+1)`` to ``n/(k+1)`` where ``n`` is the number of
        input lists, so scores are always ``> 0``.
    """
    scores: dict[str, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

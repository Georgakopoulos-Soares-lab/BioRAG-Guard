"""Query-time defence: the defended vote.

``donor_cap``, ``weight`` and ``abstain`` exist only here -- there is no
ingest-time or scan-time form of them -- and in the research grid the vote
is where the insert-arm attacks are actually defeated.
"""

from .policy import VotePolicy, admit, dedup_keep, defended_vote, label_agreement, survivors
from .weights import SCORE_EPS, WEIGHT_SCHEMES, vote_weights

__all__ = [
    "SCORE_EPS", "WEIGHT_SCHEMES", "VotePolicy", "admit", "dedup_keep",
    "defended_vote", "label_agreement", "survivors", "vote_weights",
]

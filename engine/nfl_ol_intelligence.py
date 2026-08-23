"""NFL offensive line replacement intelligence (audit/model context layer).

Activates only when an OL starter is unavailable. Does not alter healthy
offensive line ratings.
"""

from dataclasses import dataclass, asdict
from typing import Any

OL_WEIGHTS = {"LT":0.30,"C":0.25,"RT":0.20,"LG":0.125,"RG":0.125}

@dataclass
class OLReplacementContext:
    position: str
    starter: str
    replacement: str
    rating_drop: float
    experience_credit: float
    effective_drop: float
    opponent_pass_rush_modifier: float
    final_impact: float

def calculate_ol_replacement_impact(position, starter_rating, replacement_rating,
                                    experience_credit=0.0,
                                    opponent_pass_rush_modifier=0.0):
    weight=OL_WEIGHTS.get(position,0.15)
    drop=max(0,float(starter_rating)-float(replacement_rating))
    effective=max(0,drop-float(experience_credit))
    impact=-(effective*0.12*weight)
    impact += float(opponent_pass_rush_modifier)
    return max(-2.5,min(0,impact))

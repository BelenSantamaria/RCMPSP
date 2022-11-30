from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class Solution:
    mpdi: int
    makespan: int

    job_finishing_time: List[int]
    resource_usage: List[np.ndarray]

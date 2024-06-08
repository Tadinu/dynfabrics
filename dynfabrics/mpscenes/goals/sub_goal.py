from dataclasses import dataclass
from typing import List
from enum import Enum
from abc import abstractmethod

import numpy as np

from dynfabrics.mpscenes.common.component import MPComponent
from dynfabrics.mpscenes.common.errors import MissmatchDimensionError


class SubGoalType(Enum):
    STATIC = 0  # Note: This does not necessarily means fixed or stationary!
    ANALYTIC = 1
    SPLINE = 2


@dataclass
class SubGoalConfig:
    """Configuration dataclass for sub goal.

    This configuration class holds information about the
    the weight, accuracy required, type and position in the
    kinematic chain.

    Parameters:
    ------------

    m: int: Dimension of the sub goal
    w: float: Weight of the sub goal
    type: str: Type of the sub goal
    indices: list: Indices of a forward map to be considered
    epsilon: float: Required accuracy of the sub goal
    prime: bool: Flag for primary goal
    """

    weight: float
    type: SubGoalType
    indices: List[int]
    epsilon: float
    is_primary_goal: bool


class SubGoal(MPComponent):
    def check_dimensionality(self):
        if isinstance(self.pose(), str):
            return

    def is_primary_goal(self):
        return self._config.is_primary_goal

    def epsilon(self) -> float:
        return self._config.epsilon

    def indices(self) -> List[int]:
        return self._config.indices

    def dimension(self) -> int:
        return len(self.indices())

    def weight(self) -> float:
        return self._config.weight

    def type(self) -> SubGoalType:
        return self._config.type

    def evaluate(self, **kwargs) -> list:
        return [
            self.pose(**kwargs),
            self.velocity(**kwargs),
            self.acceleration(**kwargs)
        ]

    def position(self, **kwargs) -> np.ndarray:
        return np.zeros(3)

    def orientation(self, **kwargs) -> np.ndarray:
        return np.array([1, 0, 0, 0])

    def pose(self, **kwargs) -> np.ndarray:
        return np.eye(4)

    @abstractmethod
    def velocity(self, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def acceleration(self, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def shuffle(self):
        pass

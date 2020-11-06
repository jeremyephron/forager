from dataclasses import dataclass
from enum import Enum
import heapq

from dataclasses_json import dataclass_json
import numpy as np
from runstats import Statistics

from typing import Callable, List, Optional

from knn import utils
from knn.utils import JSONType

from .base import Reducer


class TopKReducer(Reducer):
    @dataclass_json
    @dataclass
    class ScoredResult:
        score: float
        input: JSONType
        output: JSONType

        def __lt__(self, other):
            return self.score < other.score

    def __init__(
        self, k: int, extract_func: Optional[Callable[[JSONType], float]] = None
    ) -> None:
        super().__init__()
        self.k = k
        self.extract_func = extract_func or self.extract_value
        self._top_k: List[TopKReducer.ScoredResult] = []

    def handle_result(self, input: JSONType, output: JSONType) -> None:
        result = TopKReducer.ScoredResult(self.extract_func(output), input, output)
        if len(self._top_k) < self.k:
            heapq.heappush(self._top_k, result)
        elif self._top_k[0] < result:
            heapq.heapreplace(self._top_k, result)

    def extract_value(self, output: JSONType) -> float:
        assert isinstance(output, float)
        return output

    @property
    def result(self) -> List[ScoredResult]:
        return list(reversed(sorted(self._top_k)))


class PoolingReducer(Reducer):
    class PoolingType(Enum):
        MAX = np.max
        AVG = np.mean

    def __init__(
        self,
        pool_func: PoolingType = PoolingType.AVG,
        extract_func: Optional[Callable[[JSONType], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.pool_func = pool_func
        self.extract_func = extract_func or self.extract_value
        self._results = []  # type: List[np.ndarray]

    def handle_result(self, input: JSONType, output: JSONType) -> None:
        self._results.append(self.extract_func(output))

    def extract_value(self, output: JSONType) -> np.ndarray:
        assert isinstance(output, str)
        return utils.base64_to_numpy(output)

    @property
    def result(self) -> np.ndarray:
        return self.pool_func(np.stack(self._results), axis=0)

class TrivialReducer(Reducer):
    def __init__(
        self,
        extract_func: Optional[Callable[[JSONType], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.pool_func = pool_func
        self.extract_func = extract_func or self.extract_value
        self._results = []  # type: List[np.ndarray]

    def handle_result(self, input: JSONType, output: JSONType) -> None:
        self._results.append(self.extract_func(output))

    def extract_value(self, output: JSONType) -> np.ndarray:
        assert isinstance(output, str)
        return utils.base64_to_numpy(output)

    @property
    def result(self) -> np.ndarray:
        return np.stack(self._results)

class StatisticsReducer(Reducer):
    def __init__(self, extract_func: Optional[Callable[[JSONType], float]] = None):
        super().__init__()
        self._result = Statistics()
        self.extract_func = extract_func or self.extract_value

    def handle_result(self, input: JSONType, output: JSONType) -> None:
        self._result.push(self.extract_func(output))

    def extract_value(self, output: JSONType) -> float:
        assert isinstance(output, float) or isinstance(output, int)
        return output

    @property
    def result(self) -> Statistics:
        return self._result

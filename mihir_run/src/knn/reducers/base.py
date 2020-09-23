import abc

from typing import Any

from knn.utils import JSONType


class Reducer(abc.ABC):
    @abc.abstractmethod
    def handle_result(self, input: JSONType, output: JSONType) -> None:
        pass

    @abc.abstractproperty
    def result(self) -> Any:
        pass

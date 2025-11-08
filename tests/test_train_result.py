import logging
import typing as t
from typing_extensions import Self
import torch

LOGGER = logging.getLogger(__name__)

class Result:

    def __init__(self) -> None:
        self._values = {}

    @property
    def column_names(self) -> t.List[str]:
        return list(self._values.keys())
    
    @property
    def values(self) -> t.Dict[str, t.List[float]]:
        return self._values

    def add(self, new: t.Dict[str, torch.Tensor]) -> Self:
        for name, value in new.items():
            if name not in self._values:
                self._values[name] = list()
            self._values[name].append(value.item())
        return self

    def __add__(self, new: t.Dict[str, torch.Tensor]) -> Self:
        return self.add(new)

    def __radd__(self, new: t.Dict[str, torch.Tensor]) -> Self:
        return self.add(new)

    def __getitem__(
        self,
        index: t.Union[int, str],
    ) -> t.Union[t.Dict[str, float], t.List[float]]:
        if isinstance(index, str):
            return self._values[index]
        value = {}
        for name in self._values:
            value[name] = self._values[name][index]
        return value


def test_result_class() -> None:
    res = Result()
    res += {"loss": torch.tensor(10), "score": torch.tensor(0.45)}
    LOGGER.info(str(res.values))

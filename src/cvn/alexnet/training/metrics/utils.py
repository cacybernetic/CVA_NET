import typing_extensions as t
import torch


class AvgMeter:
    """
    Allow to accumulate the metric value adding each value received by add and radd operators.
    From these values accumulated, we can get the average of all the values.
    """

    def __init__(
        self,
        initial_value: float=0.0,
        dtype=torch.float32,
        device: t.Union[str, torch.device] = 'cpu'
    ) -> None:
        """Method of average metter instanciation."""
        self.acc = torch.tensor(initial_value, dtype=dtype, device=device)
        self.device = device
        self.num = torch.tensor(0, dtype=torch.int32, device=device)
        self.inc = torch.tensor(1, dtype=torch.int32, device=self.device)

    def __add__(self, new_value: torch.Tensor) -> t.Self:
        """Make the operation: y = x + value."""
        self.acc = torch.add(self.acc, new_value)
        self.num = torch.add(self.num, self.inc)
        return self

    def __radd__(self, new_value: torch.Tensor) -> t.Self:
        """Make the operation: x += value."""
        return self.__add__(new_value)

    def avg(self) -> torch.Tensor:
        """Returns the average of all values added."""
        avg = torch.divide(self.acc, self.num)
        return avg

    def count(self) -> torch.Tensor:
        """Return the number of value accumulated in total."""
        return self.num

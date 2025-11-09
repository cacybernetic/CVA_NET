import logging
import torch

LOGGER = logging.getLogger(__name__)


def test_tensor_add() -> None:
    a = torch.tensor([2, 3, 1, 3])
    b = torch.tensor([1, 2, 3, 5])
    c = torch.add(a, b)
    LOGGER.info("a.mean: " + str(a.mean(dtype=torch.float32)))
    LOGGER.info("b.mean: " + str(b.mean(dtype=torch.float32)))
    LOGGER.info(str(c))

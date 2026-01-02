import time as tm
from typing import Union, Tuple
import torch
from torchinfo import summary as pytorch_summary, ModelStatistics
from .model import JEPA


def build(
    m: JEPA,
    batchs: int=1,
    img_ch: int=3,
    img_sz: int=224,
    depth: int=8,
    device: Union[str, torch.device]='cpu'
) -> Tuple[ModelStatistics, float]:
    """
    This function to make summary for the model instance received by arguments.
    """
    x_1 = torch.randn((batchs, img_ch, img_sz, img_sz)).to(device)
    x_2 = torch.randn((batchs, img_ch, img_sz, img_sz)).to(device)
    input_data = (x_1, x_2)
    start = tm.time()
    state = pytorch_summary(
        model=m, input_data=input_data, device=device, depth=depth,
        col_names=("input_size", "output_size", "num_params", "params_percent", "trainable",), verbose=0)
    end = tm.time()
    inference_time = (end - start)
    return state, inference_time

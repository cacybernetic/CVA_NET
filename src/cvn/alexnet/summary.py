import time as tm
from typing import Union, Tuple
import torch
from torchinfo import summary as pytorch_summary, ModelStatistics
from .model import AlexNet, Config


def build(
    m: AlexNet,
    c: Config,
    batchs: int=1,
    depth: int=8,
    device: Union[str, torch.device]='cpu'
) -> Tuple[ModelStatistics, float]:
    """
    This function to make summary for the model instance received by arguments.
    """
    img_ch = c.img_channels
    img_sz = c.img_size
    x = torch.randn((batchs, img_ch, img_sz, img_sz)).to(device)
    input_data = (x,)
    start = tm.time()
    state = pytorch_summary(
        model=m, input_data=input_data, device=device, depth=depth,
        col_names=("input_size", "output_size", "num_params", "params_percent", "trainable",), verbose=0)
    end = tm.time()
    inference_time = (end - start)
    return state, inference_time

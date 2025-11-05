import time as tm
from torch import nn
from torchinfo import summary


def print_model_summary(
    model: nn.Module,
    input_data: tuple,
    depth: int=8,
    device=None
):
    """
    This function to make summary for the model instance received
    by arguments.
    """
    start = tm.time()
    state = summary(
        model=model, input_data=input_data, device=device, depth=depth,
        col_names=(
            "input_size", "output_size", "num_params", "params_percent",
            "trainable",
        )
    )
    end = tm.time()
    inference_time = (end - start)
    return state, inference_time

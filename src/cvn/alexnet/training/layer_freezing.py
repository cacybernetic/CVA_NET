import re
import logging
from typing import Tuple, List
from torch import nn
from yolo.v3.model import YOLO

LOGGER = logging.getLogger(__name__)


def _find_params(pattern: str, model: nn.Module) -> Tuple[List[str], List[nn.Parameter]]:
    found_sublayers = []
    found_params = []
    for name, param in model.named_parameters():
        if re.match(pattern=pattern, string=name):
            found_sublayers.append(name)
            found_params.append(param)
    return found_sublayers, found_params


def apply(model: YOLO, layer_names: List[str]) -> YOLO:
    assert isinstance(model, YOLO), "Expected model instance: YOLO, but we got \"" + str(type(model)) + "\"."
    assert isinstance(layer_names, list), (
        "Expected list of str for `layer_names`, but we got \"" + str(type(layer_names)) + "\".")
    LOGGER.info("Layers configuration:")
    for layer_name in sorted(layer_names):
        found_sb_layers, found_params = _find_params(layer_name, model)
        for name, p in zip(found_sb_layers, found_params):
            p.requires_grad = False
            LOGGER.info("\t* Layer named \"" + str(name) + "\" is frozen.")
    return model

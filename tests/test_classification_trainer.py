import logging
import json
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset

from cva_net.alexnet import ModelFactory as AlexnetModel
from cva_net.classification.trainer import Trainer

LOGGER = logging.getLogger(__name__)


def test_trainer_class() -> None:
    torch.manual_seed(42)
    model = AlexnetModel.build()
    train_dataset = TensorDataset(
        torch.randn((100, 3, 224, 224)),
        torch.randint(0, 32, (100,), dtype=torch.int64)
    )
    test_dataset = TensorDataset(
        torch.randn((20, 3, 224, 224)),
        torch.randint(0, 32, (20,), dtype=torch.int64)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        train_dataset=train_dataset, model=model, criterion=criterion,
        optimizer=optimizer, test_dataset=test_dataset, num_epochs=20,
    )
    trainer.compile()
    results, test_results = trainer.execute()
    LOGGER.debug("results: \n" + json.dumps(results, indent=4))
    LOGGER.debug("test_results: \n" + json.dumps(test_results, indent=4))

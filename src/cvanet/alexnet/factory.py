from .model import AlexNet, Config


def alexnet(config: Config=None, **kwargs) -> AlexNet:
    if not config:
        config = Config()
    config.__dict__.update(kwargs)
    model = AlexNet(num_channels=config.num_channels, num_classes=config.num_classes, dropout=config.dropout)
    return model, config

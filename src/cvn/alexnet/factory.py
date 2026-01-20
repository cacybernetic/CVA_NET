from .model import AlexNet, Config


def alexnet(config: Config=None, **kwargs) -> AlexNet:
    if not config:
        config = Config()
    config.__dict__.update(kwargs)
    num_classes = len(config.class_names)
    model = AlexNet(num_channels=config.img_channels, num_classes=num_classes, dropout=config.dropout)
    return model, config

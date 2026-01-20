from .model import AlexNet, Config, init_weights_he


def alexnet(config: Config=None, **kwargs) -> AlexNet:
    if not config:
        config = Config()
    config.__dict__.update(kwargs)
    num_classes = len(config.class_names)
    model = AlexNet(
        num_channels=config.img_channels, img_size=config.img_size, num_classes=num_classes, dropout=config.dropout)
    model.apply(init_weights_he)
    return model, config

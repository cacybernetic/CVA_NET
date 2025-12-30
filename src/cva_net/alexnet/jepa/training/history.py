

class History:

    def __init__(self) -> None:
        self._train_mse_losses = []
        self._train_cosine_losses = []
        self._train_total_losses = []
        self._val_mse_losses = []
        self._val_cosine_losses = []
        self._val_total_losses = []

    @property
    def count(self) -> int:
        return len(self._train_total_losses)

    @property
    def train_count(self) -> int:
        return len(self._train_total_losses)

    @property
    def val_count(self) -> int:
        return len(self._val_total_losses)

    def append_train(self, mse_loss: float, cosine_loss: float, total_loss: float) -> int:
        self._train_mse_losses.append(mse_loss)
        self._train_cosine_losses.append(cosine_loss)
        self._train_total_losses.append(total_loss)
        return len(self._train_total_losses)

    def append_val(self, mse_loss: float, cosine_loss: float, total_loss: float) -> int:
        self._val_mse_losses.append(mse_loss)
        self._val_cosine_losses.append(cosine_loss)
        self._val_total_losses.append(total_loss)
        return len(self._val_total_losses)

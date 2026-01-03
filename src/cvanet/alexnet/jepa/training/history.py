from typing import Dict, Any
import matplotlib.pyplot as plt


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

    def state_dict(self) -> Dict[str, Any]:
        return {
            'train_mse_losses': self._train_mse_losses,
            'train_cosine_losses': self._train_cosine_losses,
            'train_total_losses': self._train_total_losses,
            'val_mse_losses': self._val_mse_losses,
            'val_cosine_losses': self._val_cosine_losses,
            'val_total_losses': self._val_total_losses,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._train_mse_losses = state_dict['train_mse_losses']
        self._train_cosine_losses = state_dict['train_cosine_losses']
        self._train_total_losses = state_dict['train_total_losses']
        self._val_mse_losses = state_dict['val_mse_losses']
        self._val_cosine_losses = state_dict['val_cosine_losses']
        self._val_total_losses = state_dict['val_total_losses']

    def plot(self, save_file: str='results.jpg') -> str:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(self._train_total_losses) + 1)
        # Loss totale
        axes[0].plot(epochs, self._train_total_losses, label='Training')
        axes[0].plot(epochs, self._val_total_losses, label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)
        # MSE Loss
        axes[1].plot(epochs, self._train_mse_losses, label='Training')
        axes[1].plot(epochs, self._val_mse_losses, label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('MSE Loss')
        axes[1].legend()
        axes[1].grid(True)
        # Cosine Loss
        axes[2].plot(epochs, self._train_cosine_losses, label='Training')
        axes[2].plot(epochs, self._val_cosine_losses, label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Cosine Loss')
        axes[2].set_title('Cosine Loss')
        axes[2].legend()
        axes[2].grid(True)
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()
        return save_file

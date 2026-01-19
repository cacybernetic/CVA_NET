from typing import Dict, Any
import matplotlib.pyplot as plt


class History:

    def __init__(self) -> None:
        self._train_box_losses = []
        self._train_pobj_losses = []
        self._train_noobj_losses = []
        self._train_cls_losses = []
        self._val_box_losses = []
        self._val_pobj_losses = []
        self._val_noobj_losses = []
        self._val_cls_losses = []
        self._val_cls_accuracies = []
        self._val_pobj_accuracies = []
        self._val_noobj_accuracies = []

    @property
    def count(self) -> int:
        return len(self._train_box_losses)

    @property
    def train_count(self) -> int:
        return len(self._train_box_losses)

    @property
    def val_count(self) -> int:
        return len(self._val_box_losses)

    def append_train(self, box_loss: float, pobj_loss: float, noobj_loss: float, cls_loss: float) -> int:
        self._train_box_losses.append(box_loss)
        self._train_pobj_losses.append(pobj_loss)
        self._train_noobj_losses.append(noobj_loss)
        self._train_cls_losses.append(cls_loss)
        return len(self._train_box_losses)

    def append_val(
        self, box_loss: float, pobj_loss: float, noobj_loss: float, cls_loss: float, cls_acc: float, pobj_acc: float,
        noobj_acc: float
    ) -> int:
        self._val_box_losses.append(box_loss)
        self._val_pobj_losses.append(pobj_loss)
        self._val_noobj_losses.append(noobj_loss)
        self._val_cls_losses.append(cls_loss)
        self._val_cls_accuracies.append(cls_acc)
        self._val_pobj_accuracies.append(pobj_acc)
        self._val_noobj_accuracies.append(noobj_acc)
        return len(self._val_box_losses)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'train_box_losses': self._train_box_losses,
            'train_pobj_losses': self._train_pobj_losses,
            'train_noobj_losses': self._train_noobj_losses,
            'train_cls_losses': self._train_cls_losses,
            'val_box_losses': self._val_box_losses,
            'val_pobj_losses': self._val_pobj_losses,
            'val_noobj_losses': self._val_noobj_losses,
            'val_cls_losses': self._val_cls_losses,
            'val_cls_accuracies': self._val_cls_accuracies,
            'val_pobj_accuracies': self._val_pobj_accuracies,
            'val_noobj_accuracies': self._val_noobj_accuracies,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._train_box_losses = state_dict['train_box_losses']
        self._train_pobj_losses = state_dict['train_pobj_losses']
        self._train_noobj_losses = state_dict['train_noobj_losses']
        self._train_cls_losses = state_dict['train_cls_losses']
        self._val_box_losses = state_dict['val_box_losses']
        self._val_pobj_losses = state_dict['val_pobj_losses']
        self._val_noobj_losses = state_dict['val_noobj_losses']
        self._val_cls_losses = state_dict['val_cls_losses']
        self._val_cls_accuracies = state_dict['val_cls_accuracies']
        self._val_pobj_accuracies = state_dict['val_pobj_accuracies']
        self._val_noobj_accuracies = state_dict['val_noobj_accuracies']

    def plot(self, save_file: str='results.jpg') -> str:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        epochs = range(1, len(self._train_box_losses) + 1)
        # Box Loss
        axes[0][0].plot(epochs, self._train_box_losses, label='Training')
        axes[0][0].plot(epochs, self._val_box_losses, label='Validation')
        axes[0][0].set_xlabel('Epoch')
        axes[0][0].set_ylabel('Loss')
        axes[0][0].set_title('Box Loss')
        axes[0][0].legend()
        axes[0][0].grid(True)
        # CLS Loss
        axes[0][1].plot(epochs, self._train_cls_losses, label='Training')
        axes[0][1].plot(epochs, self._val_cls_losses, label='Validation')
        axes[0][1].set_xlabel('Epoch')
        axes[0][1].set_ylabel('Loss')
        axes[0][1].set_title('Class Loss')
        axes[0][1].legend()
        axes[0][1].grid(True)
        # Presence object Loss
        axes[1][0].plot(epochs, self._train_pobj_losses, label='Training')
        axes[1][0].plot(epochs, self._val_pobj_losses, label='Validation')
        axes[1][0].set_xlabel('Epoch')
        axes[1][0].set_ylabel('Loss')
        axes[1][0].set_title('Presence Object Loss')
        axes[1][0].legend()
        axes[1][0].grid(True)
        # No object Loss
        axes[1][1].plot(epochs, self._train_noobj_losses, label='Training')
        axes[1][1].plot(epochs, self._val_noobj_losses, label='Validation')
        axes[1][1].set_xlabel('Epoch')
        axes[1][1].set_ylabel('Loss')
        axes[1][1].set_title('No Object Loss')
        axes[1][1].legend()
        axes[1][1].grid(True)
        # Class accuracy
        axes[2][0].plot(epochs, self._val_cls_accuracies)
        axes[2][0].set_xlabel('Epoch')
        axes[2][0].set_ylabel('Accuracy')
        axes[2][0].set_title('Class Accuracy')
        # axes[2][0].legend()
        axes[2][0].grid(True)
        # Pobj accuracy
        axes[2][1].plot(epochs, self._val_pobj_accuracies)
        axes[2][1].set_xlabel('Epoch')
        axes[2][1].set_ylabel('Accuracy')
        axes[2][1].set_title('Presence Object Accuracy')
        # axes[2][1].legend()
        axes[2][1].grid(True)
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()
        return save_file

from typing import Dict, Any
import matplotlib.pyplot as plt


class History:

    def __init__(self) -> None:
        self._train_entropy_losses = []
        self._train_accuracy_scores = []
        self._train_precisions = []
        self._train_recalls = []
        self._val_entropy_losses = []
        self._val_accuracy_scores = []
        self._val_precisions = []
        self._val_recalls = []

    @property
    def count(self) -> int:
        return len(self._train_entropy_losses)

    @property
    def train_count(self) -> int:
        return len(self._train_entropy_losses)

    @property
    def val_count(self) -> int:
        return len(self._val_entropy_losses)

    def append_train(self, entropy_loss: float, accuracy_score: float, precision: float, recall: float) -> int:
        self._train_entropy_losses.append(entropy_loss)
        self._train_accuracy_scores.append(accuracy_score)
        self._train_precisions.append(precision)
        self._train_recalls.append(recall)
        return len(self._train_entropy_losses)

    def append_val(self, entropy_loss: float, accuracy_score: float, precision: float, recall: float) -> int:
        self._val_entropy_losses.append(entropy_loss)
        self._val_accuracy_scores.append(accuracy_score)
        self._val_precisions.append(precision)
        self._val_recalls.append(recall)
        return len(self._val_entropy_losses)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'train_entropy_losses': self._train_entropy_losses,
            'train_accuracy_scores': self._train_accuracy_scores,
            'train_precisions': self._train_precisions,
            'train_recalls': self._train_recalls,
            'val_entropy_losses': self._val_entropy_losses,
            'val_accuracy_scores': self._val_accuracy_scores,
            'val_precisions': self._val_precisions,
            'val_recalls': self._val_recalls}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._train_entropy_losses = state_dict['train_entropy_losses']
        self._train_accuracy_scores = state_dict['train_accuracy_scores']
        self._train_precisions = state_dict['train_precisions']
        self._train_recalls = state_dict['train_recalls']
        self._val_entropy_losses = state_dict['val_entropy_losses']
        self._val_accuracy_scores = state_dict['val_accuracy_scores']
        self._val_precisions = state_dict['val_precisions']
        self._val_recalls = state_dict['val_recalls']

    def plot(self, save_file: str='results.jpg') -> str:
        _, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(self._train_entropy_losses) + 1)
        # Entropy Loss
        axes[0][0].plot(epochs, self._train_entropy_losses, label='Training')
        axes[0][0].plot(epochs, self._val_entropy_losses, label='Validation')
        axes[0][0].set_xlabel('Epochs')
        axes[0][0].set_ylabel('Entropy Loss')
        axes[0][0].set_title('Entropy Loss Evolution')
        axes[0][0].legend()
        axes[0][0].grid(True)
        # Accuracy Score
        axes[0][1].plot(epochs, self._train_accuracy_scores, label='Training')
        axes[0][1].plot(epochs, self._val_accuracy_scores, label='Validation')
        axes[0][1].set_xlabel('Epochs')
        axes[0][1].set_ylabel('Accuracy score')
        axes[0][1].set_title('Accuracy Scores Evolution')
        axes[0][1].legend()
        axes[0][1].grid(True)
        # Precision
        axes[1][0].plot(epochs, self._train_precisions, label='Training')
        axes[1][0].plot(epochs, self._val_precisions, label='Validation')
        axes[1][0].set_xlabel('Epochs')
        axes[1][0].set_ylabel('Precision (P)')
        axes[1][0].set_title('Precision Evolution')
        axes[1][0].legend()
        axes[1][0].grid(True)
        # Recall
        axes[1][1].plot(epochs, self._train_recalls, label='Training')
        axes[1][1].plot(epochs, self._val_recalls, label='Validation')
        axes[1][1].set_xlabel('Epochs')
        axes[1][1].set_ylabel('Recall (R)')
        axes[1][1].set_title('Recall Evolution')
        axes[1][1].legend()
        axes[1][1].grid(True)
        ## finalyse plot;
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()
        return save_file

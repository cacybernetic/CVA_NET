import os
import math
import random
import time as tm
from glob import glob
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision import transforms

DEFAULT_OUTPUT_DIR = 'outputs/evals'
DEFAULT_NUM_COLS = 5


class AutoAddFile:
    def __init__(self, dir_path, file_prefix, file_extension=''):
        self._dir_path = dir_path
        self._file_prefix = file_prefix
        self._file_extension = file_extension

    def get_new_file_name(self):
        file_list = glob(
            f"{self._dir_path}"
            f"/{self._file_prefix}[0-9]*.{self._file_extension}"
        )
        latest_id = len(file_list)
        return f"{self._file_prefix}{latest_id}.{self._file_extension}"


def get_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Validation/test transforms pipeline.
    Only includes necessary preprocessing without augmentation.
    """
    return transforms.Compose([
        # Convert to tensor:
        transforms.ToTensor(),
        # Resize to slightly larger than target size:
        transforms.Resize((img_size)),

        # Normalize using ImageNet statistics:
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _post_process(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Post-processing method
    ----------------------

    :param logits: [batch_size, num_classes];
    :returns: tuple of two tensors of size [batch_size,],
      the first tensor contains the class ids predicted and the second tensor contains the softmax confidences.
    """
    probs = torch.softmax(logits, dim=-1)  # [n, num_classes]
    class_ids = torch.argmax(probs, dim=-1)  # [n,]
    confidences = torch.max(probs, dim=-1).values  # [n,]
    return class_ids, confidences


class Generator:
    """
    Allows to generate predictions on some samples
    randomly chosen and draws it on a grid saved as an image.
    """

    def __init__(
        self,
        model,
        dataset,
        class_names,
        image_size,
        num_batchs=None,
        batch_size=None,
        num_rows=None,
        num_cols=None,
        fig_size=None,
        output_dir=None,
    ) -> None:
        has_get_item_func = (
            hasattr(dataset, '__getitem__') and callable(dataset.__getitem__))
        has_len_func = (
            hasattr(dataset, '__len__') and callable(dataset.__len__))
        if not has_get_item_func or not has_len_func:
            raise TypeError(
                "The instance of provided dataset is not supported."
                "Some methods like `__getitem__` and `__len__` missing.")
        self._model = model
        self._dataset = dataset
        self._class_names = class_names
        self._num_batchs = num_batchs
        self._batch_size = batch_size
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._fig_size = fig_size
        self._output_dir = output_dir
        if not self._num_batchs:
            self._num_batchs = random.randint(5, 20)
        if not self._batch_size:
            max_batch_size = int(len(self._dataset) * 0.40)
            self._batch_size = random.randint(1, max_batch_size)
        if not self._num_cols:
            self._num_cols = DEFAULT_NUM_COLS
        if not self._num_rows:
            self._num_rows = math.ceil(self._batch_size / self._num_cols)
            if self._num_rows <= 1:
                self._num_rows += 1
        if not self._fig_size:
            width = int(3.5 * self._num_cols)
            height = int(3 * self._num_rows)
            self._fig_size = (width, height)
        if not self._output_dir:
            self._output_dir = DEFAULT_OUTPUT_DIR
            if not os.path.isdir(self._output_dir):
                os.makedirs(self._output_dir)
        self.auto_add_file = AutoAddFile(self._output_dir, 'spd', 'png')
        self._times = []
        self._image_preprocess = get_transforms(image_size)
        self._pbar = tqdm(
            total=(self._num_batchs * self._num_cols * self._num_rows),
            desc="Visual Validation",
            unit=" image(s)",
            # ncols=120,  # Fixed width for consistent display;
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]',
            leave=False)  # Don't leave progress bar after completion;

    @property
    def mean_time(self):
        time_sum = sum(self._times)
        return time_sum / len(self._times)

    def _predict(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        # image = np.asarray(image)
        start_time = tm.time()
        # inputs = self._image_preprocess(image)
        inputs = image.unsqueeze(0)
        logits = self._model.forward(inputs)
        predictions, confidences = _post_process(logits)
        self._times.append(tm.time() - start_time)
        return predictions[0], confidences[0]

    def make_prediction(self, indices: torch.Tensor, show_result=False) -> None:
        plt.figure()
        fig, axs = plt.subplots(self._num_rows, self._num_cols, figsize=self._fig_size)
        # Iterate over the subplots and display random images
        # from the training dataset.
        # Choose a random index from the training dataset;
        # Make prediction with the model;
        # Display the image in the subplot;
        # Set the title of the subplot as the corresponding class name;
        # Disable the axis for better visualization.
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                image_index = indices[i*self._num_cols + j]
                sample = self._dataset[image_index]
                image = sample[0]
                ## Make prediction;
                class_id, confidence = self._predict(image)
                image = np.transpose(image, (1, 2, 0))
                axs[i, j].imshow(image)
                # sample = self._dataset[image_index]
                # class_name = self._dataset.class_names[sample[1]]
                class_name = self._class_names[class_id]
                color_name = '#00ff08' if class_id == sample[1].item() \
                    else '#ff0800'
                string_title = f"{class_name} ({int(confidence.item()*100)}%)"
                axs[i, j].set_title(string_title, color=color_name)
                axs[i, j].axis(False)
                self._pbar.update(1)
        # Set the super title of the figure;
        # Set the background color of the figure as black.
        fig.suptitle(
            (f"Random {self._num_rows * self._num_cols} images taken from dataset with their predictions."),
            fontsize=16, color="white")
        fig.set_facecolor(color='black')
        file_name = self.auto_add_file.get_new_file_name()
        file_path = os.path.join(self._output_dir, file_name)
        plt.savefig(file_path)
        if show_result:
            plt.show()
        plt.close()

    def make_predictions(self, show_result=False) -> None:
        count = self._num_rows * self._num_cols
        indices = torch.randint(0, len(self._dataset), (self._num_batchs, count))
        for i in range(self._num_batchs):
            self.make_prediction(indices[i], show_result)
        self._pbar.write(f"Prediction of {self._num_batchs} batchs of {count} samples are done.")
        self._pbar.close()

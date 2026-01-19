import os
import logging
import random
from typing import Callable, List, Tuple, Set
import numpy as np
from PIL import Image, ImageFile
import cv2 as cv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# from utils import iou_width_height as iou, non_max_suppression as nms

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp',
    '.tiff', '.tif', '.webp', '.svg', '.ico',
    '.heic', '.heif', '.raw', '.cr2', '.nef',
    '.arw', '.psd', '.ai', '.eps'
}
LOGGER = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_data_preprocessors(image_size: int, scale=1.1) -> Tuple[A.Compose, A.Compose]:
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(image_size * scale)),
            A.PadIfNeeded(
                min_height=int(image_size * scale),
                min_width=int(image_size * scale),
                border_mode=cv.BORDER_CONSTANT,
            ),
            A.RandomCrop(width=image_size, height=image_size),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv.BORDER_CONSTANT),
                    # A.IAAAffine(shear=15, p=0.5, mode="constant"),
                    A.Affine(shear=15, p=0.5, mode="constant"),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
    )
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv.BORDER_CONSTANT),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )
    return train_transforms, test_transforms


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = (boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection)
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if (
                box[0] != chosen_box[0]
                or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format,)
                    < iou_threshold
            )
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


# def calculate_iou_boxes(box1, box2):
#     """
#     Calcule l'IoU entre deux boîtes (format: x_center, y_center, width, height normalisés)
#     """
#     x1_min = box1[0] - box1[2] / 2
#     y1_min = box1[1] - box1[3] / 2
#     x1_max = box1[0] + box1[2] / 2
#     y1_max = box1[1] + box1[3] / 2
    
#     x2_min = box2[0] - box2[2] / 2
#     y2_min = box2[1] - box2[3] / 2
#     x2_max = box2[0] + box2[2] / 2
#     y2_max = box2[1] + box2[3] / 2
    
#     inter_x_min = max(x1_min, x2_min)
#     inter_y_min = max(y1_min, y2_min)
#     inter_x_max = min(x1_max, x2_max)
#     inter_y_max = min(y1_max, y2_max)
    
#     if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
#         return 0.0
    
#     inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
#     box1_area = box1[2] * box1[3]
#     box2_area = box2[2] * box2[3]
#     union_area = box1_area + box2_area - inter_area
    
#     return inter_area / union_area if union_area > 0 else 0.0


def generate_background_boxes(existing_boxes, num_bg_boxes=8, min_size=0.05, max_size=0.2, max_attempts=50):
    """
    Génère des boîtes background dans les zones sans objets.

    Args:
        existing_boxes: Liste des boîtes existantes [x, y, w, h, class].
        num_bg_boxes: Nombre de boîtes background à générer.
        min_size: Taille minimale (normalisée) des boîtes background.
        max_size: Taille maximale (normalisée) des boîtes background.
        max_attempts: Nombre maximum de tentatives pour placer une boîte.

    Returns:
        Liste de boîtes background [x, y, w, h, 0] où 0 est la classe background.
    """
    background_boxes = []
    for _ in range(max_attempts):
        if len(background_boxes) >= num_bg_boxes:
            break
        # Générer une boîte aléatoire
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)
        x = random.uniform(w/2, 1 - w/2)
        y = random.uniform(h/2, 1 - h/2)
        # Créer un tensor pour la nouvelle boîte candidate
        candidate_box = torch.tensor([[x, y, w, h]])
        # Vérifier qu'elle ne chevauche pas les objets existants
        overlap = False
        for box in existing_boxes:
            # Convertir la boîte existante en tensor
            existing_box = torch.tensor([[box[0], box[1], box[2], box[3]]])
            # Calculer l'IoU en utilisant la fonction intersection_over_union;
            iou = intersection_over_union(candidate_box, existing_box, box_format="midpoint")
            if iou.item() > 0.01:  # Seuil minimal d'overlap
                overlap = True
                break
        # Si pas de chevauchement significatif, ajouter la boîte background
        if not overlap:
            background_boxes.append([x, y, w, h, 0])  # classe 0 = background
    return background_boxes


class YOLODataset(Dataset):

    def __init__(
        self,
        image_files: List[str],
        label_files: List[str],
        anchors: List[List[float]],
        img_size: int=416,
        img_channels: int=3,
        s: List[int]=[13, 26, 52],
        # c: int=20,
        transform: Callable=None,
        add_background: bool = True,
        num_bg_boxes: int = 8,
        bg_min_size: float = 0.05,
        bg_max_size: float = 0.15
    ) -> None:
        super().__init__()
        self._image_files = image_files
        self._label_files = label_files
        self._anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales;
        self._image_size = img_size
        self._img_channels = img_channels
        self._s = s
        # self._c = c
        self._num_anchors = self._anchors.shape[0]
        self._transform = transform
        self._num_anchors_per_scale = self._num_anchors // 3
        self._ignore_iou_thresh = 0.5
        # Background;
        self._add_background = add_background
        self._num_bg_boxes = num_bg_boxes
        self._bg_min_size = bg_min_size
        self._bg_max_size = bg_max_size

    def __len__(self) -> int:
        # return 5
        return len(self._image_files)

    def __getitem__(self, index: int):
        # label_path = os.path.join(self._label_dir, self._annotations.iloc[index, 1])
        image_path = self._image_files[index]
        label_path = self._label_files[index]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image = np.array(Image.open(image_path).convert("RGB"))
        # Décaler toutes les classes existantes de +1;
        # (car background prendra l'index 0)
        for box in bboxes:
            box[4] += 1  # class_label devient class_label + 1
        ## Si on ajoute la classe background;
        if self._add_background:
            # Générer des boîtes background si l'image contient déjà des objets;
            # if len(bboxes) > 0:
            bg_boxes = generate_background_boxes(
                existing_boxes=bboxes, num_bg_boxes=self._num_bg_boxes, min_size=self._bg_min_size,
                max_size=self._bg_max_size)
            # Ajouter les boîtes background à la liste;
            bboxes.extend(bg_boxes)
        if self._transform is not None:
            augmentations = self._transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']
        targets = [torch.zeros((self._num_anchors // 3, s, s, 6)) for s in self._s]  # [p_o, x, y, w, h, c]
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self._anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self._num_anchors_per_scale  # 0, 1, 2
                anchor_on_scale = anchor_idx % self._num_anchors_per_scale  # 0, 1, 2
                s = self._s[scale_idx]
                i, j = int(s * y), int(s * x)  # x = 0.5, s = 13 --> int(6.5) = 6
                anchor_token = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_token and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = (s * x - j), (s * y - i)  # both are between [0, 1];
                    width_cell, height_cell = (width * s, height * s)  # (s=13, width=0.5, so --> 6.5, ...)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_token and iou_anchors[anchor_idx] > self._ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore this prediction;
        # bboxes = torch.tensor([[int(box[4]), 1.0, *box[:4]] for box in bboxes])
        return image, tuple(targets)

'''
def enumerate_image_files(directory_path: str, recursive: bool = True,extensions: Set[str] = None) -> List[str]:
    """
    Enumerate all image files in a directory and its subdirectories.

    This function uses os.walk() to recursively traverse the directory structure and collects files
    with image extensions.

    Args:
        directory_path: Path to the directory to search.
        recursive: If True, search in subdirectories (default: True).
        extensions: Set of image file extensions to look for. If None, uses default image extensions.

    Returns:
        List of full paths to image files found.

    Raises:
        FileNotFoundError: If directory_path doesn't exist.
        NotADirectoryError: If directory_path is not a directory.

    Example:
        >>> files = enumerate_image_files("/path/to/photos")
        >>> print(f"Found {len(files)} image files")
    """
    # Validate directory exists and is a directory;
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")
    # Default image extensions if none provided;
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    image_files = []
    # Use os.walk to traverse directory tree;
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            # Get file extension and convert to lowercase for case-insensitive comparison;
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in extensions:
                # Construct full path and add to list;
                full_path = os.path.join(dirpath, filename)
                image_files.append(full_path)
        # If not recursive, break after first level;
        if not recursive:
            # Clear dirnames to prevent os.walk from going deeper;
            dirnames.clear()
    return image_files
'''

def _is_image_file(file_name: str):
    file_name_split = file_name.split('.')
    extension = "." + file_name_split[-1]
    return extension in IMAGE_EXTENSIONS


def _file_file_name(file_name: str) -> str:
    file_name_split = file_name.split('.')
    file_name = '.'.join([x for x in file_name_split[:-1]])
    return file_name

def _map_images_to_labels(image_dir: str, image_files: List[str], label_dir: str, label_files: List[str]) -> Tuple:
    pbar = tqdm(image_files, leave=False, desc="Mapping data")
    images = []
    labels = []
    file_mapped = 0
    for image_file in pbar:
        image_fn = _file_file_name(image_file)
        # print(image_fn)
        # exit(0)
        label_fn_found = list(filter(lambda x: x.startswith(image_fn), label_files))
        if not label_fn_found:
            pbar.write("No label file found for \"" + image_fn + "\".")
            continue
        # pbar.write(str(image_file) + " --> " + str(label_fn_found))
        images.append(os.path.join(image_dir, image_file))
        labels.append(os.path.join(label_dir, label_fn_found[0]))
        file_mapped += 1
    LOGGER.info("Number of files mapped: \"" + str(file_mapped) + "\".")
    return images, labels


def _filter_data(image_files: List[str], label_files: List[str], transform: Callable) -> Tuple[List[str], List[str]]:
    filtered_image_files = []
    filtered_label_files = []
    pbar = tqdm(total=len(image_files), leave=False, desc="Filtering")
    for image_file, label_file in zip(image_files, label_files):
        try:
            image = np.array(Image.open(image_file).convert("RGB"))
            bboxes = np.roll(np.loadtxt(fname=label_file, delimiter=" ", ndmin=2), 4, axis=1).tolist()
            augmentations = transform(image=image, bboxes=bboxes)
            bboxes = augmentations['bboxes']
            # for box in bboxes:
            #     assert (
            #         0 <= float(box[0]) < 1
            #         and 0 <= float(box[1]) < 1
            #         and 0 <= float(box[2]) < 1
            #         and 0 <= float(box[3]) < 1
            #     ), "The max value overflow: " + str(box)
            filtered_image_files.append(image_file)
            filtered_label_files.append(label_file)
        except UserWarning as e:
            pass
        except Exception as e:
            pbar.write(f"Error for {label_file}: " + str(e))
        pbar.update(1)
    return filtered_image_files, filtered_label_files


def build(
    anchors: List[List[float]],
    dataset_dir: str=None,
    train_data_dir: str=None,
    val_data_dir: str=None,
    img_size: int=416,
    img_channels: int=3,
    s: List[int]=[13, 26, 52],
    batch_size: int=32,
    num_workers: int=2,
    pin_memory: bool=False,
) -> Tuple[DataLoader, DataLoader]:
    assert dataset_dir or train_data_dir, "No training dataset directory provided."
    assert dataset_dir or val_data_dir, "No validation dataset directory provided."
    train_dataset = None
    val_dataset = None
    class_names = None
    train_transform, val_transform = build_data_preprocessors(image_size=img_size)
    if train_data_dir and val_data_dir:
        train_image_dir = os.path.join(train_data_dir, 'images')
        train_label_dir = os.path.join(train_data_dir, 'labels')
        val_image_dir = os.path.join(val_data_dir, 'images')
        val_label_dir = os.path.join(val_data_dir, 'labels')
        if not os.path.isdir(train_image_dir):
            raise FileNotFoundError("No such training dataset directory at \"%s\"." % (train_image_dir,))
        if not os.path.isdir(train_label_dir):
            raise FileNotFoundError("No such training dataset directory at \"%s\"." % (train_label_dir,))
        if not os.path.isdir(val_image_dir):
            raise FileNotFoundError("No such validation dataset directory at \"%s\"." % (val_image_dir,))
        if not os.path.isdir(val_label_dir):
            raise FileNotFoundError("No such validation dataset directory at \"%s\"." % (val_label_dir,))
        train_image_files = [image_fn for image_fn in os.listdir(train_image_dir) if _is_image_file(image_fn)]
        train_label_files = [label_fn for label_fn in os.listdir(train_label_dir) if label_fn.endswith('.txt')]
        val_image_files = [image_fn for image_fn in os.listdir(val_image_dir) if _is_image_file(image_fn)]
        val_label_files = [label_fn for label_fn in os.listdir(val_label_dir) if label_fn.endswith('.txt')]
        train_images, train_labels = _map_images_to_labels(
            train_image_dir, train_image_files, train_label_dir, train_label_files)
        val_images, val_labels = _map_images_to_labels(
            val_image_dir, val_image_files, val_label_dir, val_label_files)
        ## Filtering;
        train_images, train_labels = _filter_data(train_images, train_labels,train_transform)
        val_images, val_labels = _filter_data(val_images, val_labels, val_transform)
        ## Create datasets;
        train_dataset = YOLODataset(
            image_files=train_images, label_files=train_labels, anchors=anchors, img_size=img_size,
            img_channels=img_channels, s=s, transform=train_transform)
        val_dataset = YOLODataset(
            image_files=val_images, label_files=val_labels, anchors=anchors, img_size=img_size,
            img_channels=img_channels, s=s, transform=val_transform, add_background=False)
    train_dataset_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataset_loader, val_dataset_loader, class_names

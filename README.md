
<div align="center">
  
# CVA_NET: COMPUTER VISION ANALYZING NETWORK

![](https://img.shields.io/badge/Python-3.10%2B-blue)
![](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![](https://img.shields.io/badge/TensorFlow-2.12%2B-yellow)
![](https://img.shields.io/badge/LICENSE-MIT-%2300557f)
![](https://img.shields.io/badge/version-0.1.0-green)
![](https://img.shields.io/badge/contact-dr.mokira%40gmail.com-blueviolet)

**A comprehensive computer vision framework for easy model training and deployment**

</div>

A powerful, user-friendly open-source computer vision framework that simplifies the process of training and deploying state-of-the-art neural network models for various computer vision tasks.

**Table of Contents**

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
  - [For Linux](#for-linux)
    - [OS Dependencies](#os-dependencies)
    - [Project Dependencies](#project-dependencies)
  - [For Windows](#for-windows)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Supported Models](#supported-models)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

CVA_NET is an advanced computer vision framework designed to democratize AI by making complex computer vision tasks accessible to developers, researchers, and enthusiasts. Our platform provides a unified interface for training, evaluating, and deploying various neural network architectures with minimal setup and maximum flexibility.

Key advantages:
- **Easy Training**: Simplified pipeline for training custom models
- **Multiple Architectures**: Support for CNN, R-CNN, YOLO, Transformers, and more
- **Production Ready**: Easy model export and deployment
- **Extensible**: Modular design for custom implementations

## Features

### Model Training & Architecture
- **Image Classification**: ResNet, EfficientNet, Vision Transformers
- **Object Detection**: Faster R-CNN, YOLO variants, SSD
- **Instance Segmentation**: Mask R-CNN, SOLO
- **Multi-task Learning**: Simultaneous training for multiple objectives
- **Transfer Learning**: Pre-trained models and fine-tuning capabilities

### Data Processing
- **Advanced Augmentation**: Comprehensive data augmentation pipeline
- **Dataset Management**: Support for popular datasets (COCO, ImageNet, Pascal VOC)
- **Data Loaders**: Optimized data loading for performance
- **Preprocessing**: Built-in normalization and transformation

### Training & Evaluation
- **Distributed Training**: Multi-GPU training support
- **Hyperparameter Optimization**: Automated tuning capabilities
- **Visualization**: Real-time training metrics and loss visualization
- **Model Checkpointing**: Automatic saving and resuming

### Deployment
- **Model Export**: Export to ONNX, TorchScript, TensorFlow SavedModel
- **Inference API**: Simple inference interface
- **Web Demo**: Built-in web interface for model testing

## Installation

To install the project, make sure you have Python 3.8 or later version
and `pip` installed on your machine. And then run the following command lines.

### For Linux

```bash
git clone https://github.com/cacybernetic/CVA_NET;
cd CVA_NET;
sudo rm -r .git;
git init;  # To create a new instance of git repository
```

#### OS dependences

##### Ubuntu
Open your terminal and run following command lines
to add the deadsnakes PPA to your system:

```sh
sudo apt update;
sudo apt install software-properties-common -y;
sudo add-apt-repository ppa:deadsnakes/ppa -y

```

Refresh your package list to include the deadsnakes PPA
and then install Python 3.10:

```sh
sudo apt update;
sudo apt install python3.10;
python3.10 --version
```

> **NOTE**: Do not change the default Python version of Ubuntu,
> as it may break system tools that depend on it.

##### Debian or Kali

In first, install the following dependences on your computer.

```sh
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
```

And then, we can run the following command to install `pyenv`
directly via APT on your computer.

```sh
sudo apt install pyenv
```

Or run the following command lines, to clone and install
`pyenv` from its souce code.

```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv;
 
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc;
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc;
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc;
echo 'eval "$(pyenv init -)"' >> ~/.bashrc;
source ~/.bashrc;
```

Now, runing the following command line, we can use `pyenv`
to install the version of Python what we want to install.

```sh
pyenv install 3.10.18;  # Here, we install Python 3.10.18.
sudo ln -s $HOME/.pyenv/versions/3.10.18/bin/python3 /usr/local/bin/python3.10
```

#### Project dependences

1. `sudo apt install cmake python3-venv` Install *Cmake* and *Virtual env*;
2. `python3 -m venv .venv` create a virtual env into directory
named `env`;
3. `source .venv/bin/activate` activate the virtual environment named `.venv`;
4. `make install` install the requirements of this package;
5. `make dev_install` or `pip install -e .` install the package in dev mode
in virtual environment;
6. Run `make test` or `pytest` to execute the unit test scripts located
at `tests` directory.

### For Windows

```bash
git clone https://github.com/cacybernetic/CVA_NET
```

```bash
cd CVA_NET
```

And then, delete the hidden directory named `.git` located at the root
of the directory project.

And then,

1. Install python for windows;
2. Open your command prompt;
3. Run `python -m venv .venv` to create a virtual env into directory
named `.venv`;
4. Run `.venv\Scripts\activate` to activate the virtual environment;
5. Run `pip install -r requirements.txt` to install the requirements
of this package or project;
6. Run `pip install -e .` install the package in dev mode in virtual
environment;
7. `pytest` run the unit test scripts located at `tests` directory.

---

## Quick Start

```python
from cva_net import CVATrainer
from cva_net.models import ResNetClassifier
from cva_net.datasets import ImageFolderDataset

# Initialize dataset
dataset = ImageFolderDataset(
    data_dir="./data/train",
    image_size=(224, 224),
    augment=True
)

# Initialize model
model = ResNetClassifier(
    num_classes=10,
    backbone="resnet50",
    pretrained=True
)

# Setup trainer
trainer = CVATrainer(
    model=model,
    train_dataset=dataset,
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

# Start training
trainer.train()

# Evaluate model
metrics = trainer.evaluate()
print(f"Validation Accuracy: {metrics['accuracy']:.2f}%")
```

## Usage Examples

### Object Detection
```python
from cva_net import DetectionTrainer
from cva_net.models import FasterRCNN

model = FasterRCNN(
    num_classes=80,
    backbone="resnet50",
    pretrained_backbone=True
)

trainer = DetectionTrainer(model=model)
trainer.setup_training(
    coco_annotation_path="./annotations/instances_train2017.json",
    image_dir="./train2017"
)
trainer.train(epochs=100)
```

### Custom Training Pipeline
```python
from cva_net import TrainingConfig
from cva_net.trainers import create_trainer

config = TrainingConfig(
    model_name="efficientnet_b0",
    task="classification",
    num_classes=1000,
    batch_size=64,
    learning_rate=0.01,
    epochs=100,
    mixed_precision=True
)

trainer = create_trainer(config)
trainer.fit()
trainer.export_model("my_model.onnx")
```

## Supported Models

### Classification
- ResNet, ResNeXt
- EfficientNet, EfficientNetV2
- Vision Transformers (ViT, DeiT)
- MobileNet, ShuffleNet

### Detection
- Faster R-CNN
- YOLO (v5, v8, v11)
- SSD
- RetinaNet

### Segmentation
- Mask R-CNN
- U-Net
- DeepLab
- FCN

### Advanced Architectures
- Swin Transformers
- DETR
- ConvNeXt
- MLP-Mixer

## Tests

Run the test suite to verify installation:

```bash
# Run all tests
make test

```

## Contributing

We welcome contributions from the community! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Setup
```bash
make dev_install    # Install development dependencies
make lint           # Run code formatting
make type-check     # Run static type checking
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or support:

- **Maintainer**: CONSOLE ART CYBERNETIC
- **Email**: dr.mokira@gmail.com
- **GitHub**: [CA CYBERNETIC](https://github.com/cacybernetic)
- **Documentation**: [Read the Docs](https://cva-net.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/mokira3d48/cva_net/issues)

---

<div align="center">

**Start building intelligent vision systems today! 🚀**

</div>

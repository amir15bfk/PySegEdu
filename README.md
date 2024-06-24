# PySegEdu


<a href="http://colab.research.google.com/github/amir15bfk/PySegEdu/blob/main/PySegEdu_Colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>  

<a href="https://www.kaggle.com/code/mohamedamirbenbachir/pysegedu-kaggle"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" height=22.5></a>

PySegEdu is an advanced Python library designed for segmentation tasks, offering state-of-the-art models and easy-to-use interfaces for training, testing, and evaluating segmentation models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

PySegEdu provides a comprehensive framework for training and evaluating segmentation models on various datasets. The library includes several advanced models such as UNet, DoubleUNet, and FCBFormer, equipped with modern techniques like multi-scale feature extraction and attention mechanisms.

## Features

- Easy setup and configuration
- Multiple advanced segmentation models
- Support for popular datasets
- Comprehensive experiment management
- Detailed performance evaluation and reporting
- Customizable and extensible

## Installation

To install PySegEdu, clone the repository and install the required dependencies:

```bash
git clone https://github.com/amir15bfk/PySegEdu.git
cd PySegEdu
pip install -r requirements.txt
```

## Usage

Follow these steps to use PySegEdu for your segmentation tasks:

### 1. Clone the Project

```bash
git clone https://github.com/amir15bfk/PySegEdu.git
cd PySegEdu
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Data

```python
from utils import download
download.download()
```

### 4. Choose a Model

```python
from models import fcbformer
model = fcbformer.FCBFormer()
```

### 5. Configure an Experiment

```python
from experiments.experiment_runner import SegmentationExperiment

experiment = SegmentationExperiment(
    exp_name="352 100ep",
    dataset="B",  # B for Kvasir and CVC
    model=model,
    load=True,
    model_source="Trained_models/FCBFormer_B_352_100ep_last.pt",
    root="./data",
    size=(352, 352),
    epochs=50,
    batch_size=6,
    num_workers=0,
    lr=1e-4,
    lrs=True,
    lrs_min=1e-6,
    mgpu=False,
    seed=42
)
```

### 6. Train the Model

```python
experiment.run_experiment()
```

### 7. Test the Model

```python
experiment.report(plot=True)
```

## Models

PySegEdu includes several state-of-the-art models for segmentation:

- **Fully Convolutional Networks (FCN)**
  - FCN-32s
  - FCN-16s
  - FCN-8s
- **U-Net Variants**
  - UNet
  - UNet++
  - Res_UNet
  - Ringed_Res_UNet
  - Double UNet
- **Region-based Methods**
  - Mask-RCNN
  - DeepLab
  - DeepLabV1
- **Transformer-based Methods**
  - FCBFormer
  - DUCK_Net

## Datasets

Supported datasets include:

- **Kvasir-SEG**
- **CVC-ClinicDB**

## Experiments

You can configure and run experiments with various settings and parameters. For detailed steps, refer to the [Usage](#usage) section above.

## Results

The library provides detailed reports on the performance of the models, including precision, recall, F1 score, Dice score, and mean Intersection over Union (mIoU).

## Contributing

We welcome contributions to PySegEdu! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more detailed documentation and examples, please refer to the project's [Wiki](https://github.com/amir15bfk/PySegEdu/wiki) or [Issues](https://github.com/amir15bfk/PySegEdu/issues) for any questions or support.


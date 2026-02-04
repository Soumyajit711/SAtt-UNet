# SleekNet Repository

This repository contains the refactored code for the SleekNet project, originally from a Jupyter notebook. It implements a U-Net-like architecture with MobileNetV2 backbone and custom attention mechanisms for segmentation tasks (specifically ISIC2016).

<img width="735" height="643" alt="image" src="https://github.com/user-attachments/assets/187063e0-4035-4bf4-8b2f-b7f3c06215e2" />


## Directory Structure

- `data/`: Dataset loading logic (`dataset.py`)
- `models/`: Model architecture definition
    - `network.py`: Main model class `SAtt-UNet`
    - `encoder.py`: Encoder components
    - `decoder.py`: Decoder components
    - `blocks.py`: Basic building blocks (MSDBlock, etc.)
    - `correlation.py`: Attention mechanisms
- `utils/`: Utility functions
    - `metrics.py`: Evaluation metrics (IoU, Dice, Accuracy, etc.)
    - `loss.py`: Custom loss functions
    - `visualization.py`: Visualization helpers
- `train.py`: Training loop implementation
- `main.py`: Entry point for training
- `requirements.txt`: Project dependencies

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

To train the model, use `main.py` with the appropriate arguments pointing to your data directories:

```bash
python main.py --images_dir /path/to/images --masks_dir /path/to/masks --epochs 300 --batch_size 8
```

Optional arguments:
- `--val_images_dir`, `--val_masks_dir`: Explicit validation set paths. If not provided, a random split is used.
- `--save_path`: Path to save the best model (default: `best_model.pth`)
- `--lr`: Learning rate (default: `1e-4`)
- `--num_workers`: Number of data loading workers (default: `4`)

### Example

```bash
python main.py --images_dir ./ISIC2016_Task1_Training_Input --masks_dir ./ISIC2016_Task1_Training_GroundTruth
```

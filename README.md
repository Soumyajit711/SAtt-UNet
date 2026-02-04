# SAtt-UNet Repository

**This repository contains the refactored code for the SAtt-UNet project, originally from a Jupyter notebook. It implements a U-Net-like architecture with a lightweight backbone and custom attention mechanisms for segmentation tasks (specifically ISIC2016).**

**Overall architecture of the proposed SAtt-UNet model. The network follows an encoder–decoder U-shaped design with multi-level skip connections, and the final segmentation mask is generated through a convolutional prediction head.**
<img width="735" height="643" alt="image" src="https://github.com/user-attachments/assets/187063e0-4035-4bf4-8b2f-b7f3c06215e2" />


**The Dsc metric on the ISIC16 dataset compared to other methods with parameters (M) and FLOPs (G).**
<img width="516" height="290" alt="image" src="https://github.com/user-attachments/assets/7cda66fb-93f9-4d29-9fe8-36b6b3e3eea5" />


**Feature maps extracted at different stages of the SAtt-UNet architecture, from the Encoder (downsampling) to the Decoder (upsampling). These visualizations show how the model progressively processes and refines the information to arrive at the final segmentation.**
<img width="733" height="158" alt="image" src="https://github.com/user-attachments/assets/6d2bf389-569a-4478-b067-f27b25995f67" />


**Qualitative results of lesion segmentation with predicted masks (green) overlaid on input images, highlighting boundaries (red) and misclassified regions (blue) for all datasets.**
<img width="734" height="316" alt="image" src="https://github.com/user-attachments/assets/ff301ca5-3186-4e33-8b4c-a693bef75fe2" />


**Qualitative comparison of ground-truth masks, predicted masks, and overlay visualizations for different training–testing dataset combinations, illustrating the generalization capability of the proposed model under domain shifts.**
<img width="729" height="411" alt="image" src="https://github.com/user-attachments/assets/c5cc726c-4d60-4769-8d8c-77166a9d7725" />


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

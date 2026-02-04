# **Abstract**
The increasing demand for quick and efficient skin cancer detection has grown, making medical image segmentation a vital tool for early diagnosis. While many advanced models, particularly those based on the U-Net architecture, often come with a large number of parameters, which limits their use on devices with less computational resources. To overcome this challenge, this paper introduces a lightweight scale-aware attentive UNet model, called SAtt-UNet, for the semantic segmentation of skin cancer images. The main aim is to create a highly accurate model that does not require immense computational overhead. In the encoder, this framework incorporates a multi-scale pyramid pooling block that grinds in on the most relevant attributes, followed by a correlation-guided feature refinement block. The decoder, on the other hand, employs a multi-scale context aggregation method with cross-attention between different feature levels. This allows the model to accurately restore the spatial information and accurately segment both broad lesion areas and finer details at their edges. Extensive testing on ISIC16, PH2, and ISIC18 datasets shows that SAtt-UNet achieves high-quality results, with Dice coefficient scores of 91.97%, 94.68%, and 88.10%, respectively. This ensures that it is possible to achieve notable performance with a significantly less parameterized model, offering a practical and accessible solution for clinical settings where computational resources are often limited.


<img width="735" height="643" alt="image" src="https://github.com/user-attachments/assets/187063e0-4035-4bf4-8b2f-b7f3c06215e2" />
**Fig 1. Overall architecture of the proposed SAtt-UNet model. The network follows an encoder–decoder U-shaped design with multi-level skip connections, and the final segmentation mask is generated through a convolutional prediction head.**


<img width="516" height="290" alt="image" src="https://github.com/user-attachments/assets/7cda66fb-93f9-4d29-9fe8-36b6b3e3eea5" />
**Fig 2. The Dsc metric on the ISIC16 dataset compared to other methods with parameters (M) and FLOPs (G).**


<img width="733" height="158" alt="image" src="https://github.com/user-attachments/assets/6d2bf389-569a-4478-b067-f27b25995f67" />
**Fig 3. Feature maps extracted at different stages of the SAtt-UNet architecture, from the Encoder (downsampling) to the Decoder (upsampling). These visualizations show how the model progressively processes and refines the information to arrive at the final segmentation.**


<img width="734" height="316" alt="image" src="https://github.com/user-attachments/assets/ff301ca5-3186-4e33-8b4c-a693bef75fe2" />
**Fig 4. Qualitative results of lesion segmentation with predicted masks (green) overlaid on input images, highlighting boundaries (red) and misclassified regions (blue) for all datasets.**


<img width="729" height="411" alt="image" src="https://github.com/user-attachments/assets/c5cc726c-4d60-4769-8d8c-77166a9d7725" />
**Fig 5. Qualitative comparison of ground-truth masks, predicted masks, and overlay visualizations for different training–testing dataset combinations, illustrating the generalization capability of the proposed model under domain shifts.**


## Directory Structure

- `data/`: Dataset loading logic (`dataset.py`)
- `models/`: Model architecture definition
    - `network.py`: Main model class
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

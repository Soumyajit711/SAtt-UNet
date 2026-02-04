# **Abstract**
The increasing demand for quick and efficient skin cancer detection has grown, making medical image segmentation a vital tool for early diagnosis. While many advanced models, particularly those based on the U-Net architecture, often come with a large number of parameters, which limits their use on devices with less computational resources. To overcome this challenge, this paper introduces a lightweight scale-aware attentive UNet model, called SAtt-UNet, for the semantic segmentation of skin cancer images. The main aim is to create a highly accurate model that does not require immense computational overhead. In the encoder, this framework incorporates a multi-scale pyramid pooling block that grinds in on the most relevant attributes, followed by a correlation-guided feature refinement block. The decoder, on the other hand, employs a multi-scale context aggregation method with cross-attention between different feature levels. This allows the model to accurately restore the spatial information and accurately segment both broad lesion areas and finer details at their edges. Extensive testing on ISIC16, PH2, and ISIC18 datasets shows that SAtt-UNet achieves high-quality results, with Dice coefficient scores of 91.97%, 94.68%, and 88.10%, respectively. This ensures that it is possible to achieve notable performance with a significantly less parameterized model, offering a practical and accessible solution for clinical settings where computational resources are often limited.


<img width="735" height="643" alt="image" src="https://github.com/user-attachments/assets/187063e0-4035-4bf4-8b2f-b7f3c06215e2" />

**Fig 1. Overall architecture of the proposed SAtt-UNet model. The network follows an encoder–decoder U-shaped design with multi-level skip connections, and the final segmentation mask is generated through a convolutional prediction head.**


<img width="516" height="290" alt="image" src="https://github.com/user-attachments/assets/7cda66fb-93f9-4d29-9fe8-36b6b3e3eea5" />

**Fig 2. The Dsc metric on the ISIC16 dataset compared to other methods with parameters (M) and FLOPs (G).**


**Tab 1. Quantitative evaluation of methods on ISIC16, PH2, and ISIC18 datasets. All scores are reported in % (mean ± std).**

| Dataset | Params (M) | FLOPs (G) | Dice (DSC)   | IoU          | Accuracy     | Sensitivity  | Specificity  |
| ------- | ---------- | --------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| ISIC16  | 4.17       | 2.70      | 91.97 ± 0.04 | 85.28 ± 0.05 | 95.81 ± 0.08 | 91.09 ± 0.05 | 97.55 ± 0.06 |
| PH2     | 4.17       | 2.70      | 94.68 ± 0.08 | 89.90 ± 0.06 | 96.60 ± 0.10 | 94.91 ± 0.06 | 97.27 ± 0.07 |
| ISIC18  | 4.17       | 2.70      | 88.10 ± 0.10 | 79.70 ± 0.12 | 93.46 ± 0.10 | 92.65 ± 0.11 | 95.02 ± 0.09 |


<img width="733" height="158" alt="image" src="https://github.com/user-attachments/assets/6d2bf389-569a-4478-b067-f27b25995f67" />

**Fig 3. Feature maps extracted at different stages of the SAtt-UNet architecture, from the Encoder (downsampling) to the Decoder (upsampling). These visualizations show how the model progressively processes and refines the information to arrive at the final segmentation.**


<img width="734" height="316" alt="image" src="https://github.com/user-attachments/assets/ff301ca5-3186-4e33-8b4c-a693bef75fe2" />

**Fig 4. Qualitative results of lesion segmentation with predicted masks (green) overlaid on input images, highlighting boundaries (red) and misclassified regions (blue) for all datasets.**


**Tab 2. Quantitative results for the proposed model trained and tested across ISIC16, ISIC18, and PH2 datasets. All scores are reported in % (mean ± std).**

| Training Dataset | Testing: ISIC16 (DSC) | Testing: ISIC16 (IoU) | Testing: PH2 (DSC) | Testing: PH2 (IoU) | Testing: ISIC18 (DSC) | Testing: ISIC18 (IoU) |
| ---------------- | --------------------- | --------------------- | ------------------ | ------------------ | --------------------- | --------------------- |
| **ISIC16**       | 91.97 ± 0.04          | 85.28 ± 0.05          | 92.00 ± 0.03       | 85.32 ± 0.05       | 87.28 ± 0.07          | 78.17 ± 0.10          |
| **PH2**          | 80.77 ± 0.15          | 69.98 ± 0.17          | 94.68 ± 0.08       | 89.90 ± 0.06       | 80.33 ± 0.11          | 68.52 ± 0.14          |
| **ISIC18**       | 92.07 ± 0.03          | 85.32 ± 0.06          | 92.25 ± 0.06       | 85.78 ± 0.03       | 88.10 ± 0.10          | 79.70 ± 0.12          |


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

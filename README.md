# Self-Supervised Learning for Satellite Change Detection

This project investigates self-supervised pretraining methods for remote sensing change detection.

## Methods

- Baseline CNN
- Swin Transformer (ImageNet pretrained)
- Swin + SimSiam
- Swin + SimMIM

## Datasets

### Pretraining
- EuroSAT

### Change Detection
- LEVIR-CD

## Results

| Model | IoU | F1 |
|---------|---------:|---------:|
| CNN Baseline | 0.2837 | 0.3914 |
| Swin (ImageNet) | 0.5999 | 0.7015 |
| Swin + SimSiam | 0.6343 | 0.7292 |
| Swin + SimMIM | 0.6341 | 0.7324 |

## Installation

```bash
pip install -r requirements.txt
```

## Training

### SimSiam

```bash
python -m simsiam.train_simsiam
```

### SimMIM

```bash
python -m mim.train_mim
```

### Change Detection

```bash
python -m change_detection.train_swin_simsiam
python -m change_detection.train_swin_mim
```

## Evaluation

```bash
python -m change_detection.evaluate_swin_simsiam
python -m change_detection.evaluate_swin_mim
```

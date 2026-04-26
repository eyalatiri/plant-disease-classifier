# 🌿 Plant Disease Classifier — ResNet18 Fine-tuned on PlantVillage
<img width="1095" height="639" alt="Capture d&#39;écran 2026-04-25 212336" src="https://github.com/user-attachments/assets/a17ecda3-811a-46c3-b85f-03b3391ae42f" />

A deep learning model that classifies **38 plant diseases** across 14 crop species with **99.07% accuracy**, built using transfer learning on a pretrained ResNet18 backbone.

---

## Results

| Metric | Score |
|---|---|
| Overall Accuracy | **99.07%** |
| Macro F1 Score | 98.61% |
| Weighted F1 Score | 99.07% |
| Perfect classes (F1 = 1.0) | 11 / 38 |
| Classes above 99% F1 | 28 / 38 |
| Dataset size | 54,305 images |
| Classes | 38 diseases across 14 crops |

---

## Demo
https://huggingface.co/spaces/eyalatiri/Plant-disease-classifier 

> Upload a leaf image → get instant disease diagnosis with confidence scores

```
Tomato Early Blight     94.3%  ████████████
Tomato Late Blight       4.1%  ██
Tomato Bacterial Spot    1.2%  █
```

---

## Methodology

This project applies **progressive fine-tuning** — a staged unfreezing strategy that systematically unlocks pretrained layers while protecting the knowledge already encoded in them.

### Training Journey

| Stage | Strategy | Val Accuracy | Key Change |
|---|---|---|---|
| Model 1 | Frozen backbone, train head only | 91.75% | Baseline |
| Model 2 | Unfreeze layer4 + head | 98.25% | +6.5% jump |
| Model 3 | Add dropout + weight decay + augmentation | 98.64% | Overfitting reduced |
| Model 4 | Lower LR + ReduceLROnPlateau scheduler | **99.02%** | Converged cleanly |

### Key Techniques
- **Transfer learning** from ImageNet pretrained ResNet18
- **Progressive unfreezing** — layer4 unlocked first, then fine-tuned end-to-end
- **Regularization** — Dropout(0.4) + weight decay (1e-4) to prevent overfitting
- **Learning rate scheduling** — ReduceLROnPlateau halves LR on plateau
- **Early stopping** — saves best checkpoint, stops when val accuracy stagnates
- **Data augmentation** — random flips, rotation, color jitter on training set only

---

## Dataset

**PlantVillage Dataset** — 54,305 leaf images across 38 disease/healthy classes.

- Train split: 80% (43,444 images)
- Validation split: 20% (10,861 images)
- Fixed random seed (42) for reproducibility
- Source: [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### Crops covered
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## Model Architecture

```
ResNet18 (ImageNet pretrained)
    └── layer1 [frozen]
    └── layer2 [frozen]
    └── layer3 [frozen]
    └── layer4 [fine-tuned]
    └── fc: Dropout(0.4) → Linear(512 → 38)
```

- **Trainable params:** 8,413,222
- **Frozen params:** 2,782,784
- **Input:** 224×224 RGB images, ImageNet normalized

---

## Project Structure

```
├── plantdisease.ipynb       # Full training notebook
├── best_model.pth           # Best model weights (99.02% val acc)
├── requirements.txt         # Dependencies
└── README.md
```

---

## Quickstart

### Install dependencies
```bash
pip install torch torchvision gradio pillow
```

### Run inference on a single image
```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, 38)
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Predict
image = Image.open('leaf.jpg').convert('RGB')
tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    top3 = probs.topk(3)

for prob, idx in zip(top3.values, top3.indices):
    print(f"{class_names[idx]}: {prob*100:.1f}%")
```

### Launch Gradio demo
```python
import gradio as gr

def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top5 = probs.topk(5)
    return {class_names[i]: float(p)
            for i, p in zip(top5.indices, top5.values)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=5),
    title='🌿 Plant Disease Classifier',
    description='Upload a leaf image to detect disease'
).launch()
```

---

## Per-class Results (selected)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Apple scab | 1.000 | 0.969 | 0.984 |
| Corn Cercospora leaf spot | 0.885 | 0.895 | 0.890 |
| Grape Black rot | 0.992 | 0.996 | 0.994 |
| Orange Huanglongbing | 1.000 | 1.000 | 1.000 |
| Tomato Early blight | 0.938 | 0.991 | 0.964 |
| Tomato Target spot | 0.969 | 0.920 | 0.944 |
| Tomato Yellow Leaf Curl Virus | 0.999 | 0.997 | 0.998 |

Full classification report available in the notebook.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=3.0.0
pillow>=9.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

---

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [PlantVillage Dataset](https://arxiv.org/abs/1511.08060) — Hughes & Salathé, 2015
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

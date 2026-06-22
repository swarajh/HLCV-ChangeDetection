import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from change_detection.dataset import LevirDataset
from change_detection.baseline_model import SiameseBaseline


# -----------------------
# Dataset
# -----------------------

dataset = LevirDataset(
    "datasets/LEVIR-CD/val"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)


# -----------------------
# Load trained model
# -----------------------

model = SiameseBaseline()

model.load_state_dict(
    torch.load(
        "checkpoints/baseline.pth",
        map_location="cpu"
    )
)

model.eval()


# -----------------------
# Get one sample
# -----------------------

img_a, img_b, mask = next(iter(loader))


# -----------------------
# Prediction
# -----------------------

with torch.no_grad():

    prediction = model(
        img_a,
        img_b
    )

    prediction = torch.sigmoid(
        prediction
    )

    print(
        "Prediction statistics"
    )

    print(
        "Min:",
        prediction.min().item()
    )

    print(
        "Max:",
        prediction.max().item()
    )

    print(
        "Mean:",
        prediction.mean().item()
    )


# -----------------------
# Thresholded prediction
# -----------------------

binary_prediction = (
    prediction > 0.5
).float()


# -----------------------
# Visualization
# -----------------------

fig, ax = plt.subplots(
    1,
    5,
    figsize=(20,5)
)

# Before image

ax[0].imshow(
    img_a[0].permute(1,2,0)
)

ax[0].set_title(
    "Before"
)

ax[0].axis("off")


# After image

ax[1].imshow(
    img_b[0].permute(1,2,0)
)

ax[1].set_title(
    "After"
)

ax[1].axis("off")


# Ground truth

ax[2].imshow(
    mask[0,0],
    cmap="gray"
)

ax[2].set_title(
    "Ground Truth"
)

ax[2].axis("off")


# Raw probability map

ax[3].imshow(
    prediction[0,0],
    cmap="gray"
)

ax[3].set_title(
    "Probability Map"
)

ax[3].axis("off")


# Thresholded prediction

ax[4].imshow(
    binary_prediction[0,0],
    cmap="gray"
)

ax[4].set_title(
    "Binary Prediction"
)

ax[4].axis("off")


plt.tight_layout()
plt.show()
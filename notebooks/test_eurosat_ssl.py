from simsiam.eurosat_dataset import (
    EuroSATSimSiamDataset
)

dataset = EuroSATSimSiamDataset(
    "datasets/EuroSAT"
)

print(
    "Dataset Size:",
    len(dataset)
)

view1, view2 = dataset[0]

print(
    view1.shape
)

print(
    view2.shape
)
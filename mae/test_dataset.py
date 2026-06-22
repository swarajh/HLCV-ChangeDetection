from mae.eurosat_dataset import EuroSATMAEDataset

dataset = EuroSATMAEDataset(
    "datasets/EuroSAT"
)

print(len(dataset))

img = dataset[0]

print(img.shape)
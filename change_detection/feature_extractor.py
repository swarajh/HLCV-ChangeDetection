import os
import sys
import argparse
import logging
import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simsiam.eurosat_dataset import EuroSATSimSiamDataset

# def parse_args():
#     parser = argparse.ArgumentParser(description="Extract and Save Backbone Features")
#     parser.add_argument("--data_path", 
#                         type=str, 
#                         required=True, 
#                         help="Path to EuroSAT")
#     parser.add_argument("--model_name", 
#                         type=str, 
#                         default="swin_tiny_patch4_window7_224")
#     parser.add_argument("--batch_size", 
#                         type=int, 
#                         default=128, 
#                         help="Batch size for extraction")
#     parser.add_argument("--num_workers", 
#                         type=int, 
#                         default=4)
#     parser.add_argument("--output_dir",
#                          type=str,
#                         default="precomputed_features")
#     return parser.parse_args()

def main(data_path=None,
    model_name="swin_tiny_patch4_window7_224",
    batch_size=128,
    num_workers=4,
    output_dir="features",
    ):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the output file already exists
    save_path = os.path.join(output_dir, f"features_{model_name.replace('/', '_')}.pt")
    if os.path.exists(save_path):
        print(f"Features already exist at: {save_path}. Skipping extraction.")
        return  # Exit the script if the file already exists

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Dataset...") 
    dataset = EuroSATSimSiamDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Loading Frozen Backbone: {model_name}...")
    # num_classes=0 removes the classification head, returning the raw 1D feature vector
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
        
    model = model.to(device)
    model.eval()  # Freeze dropout/batchnorm

    all_features_view1 = []
    all_features_view2 = []

    print("Extracting features... This only happens ONCE!")
    with torch.no_grad():  # NO gradients! Super fast and uses very little VRAM.
        for view1, view2 in tqdm(loader, desc="Extracting"):
            view1, view2 = view1.to(device), view2.to(device)
            
            feat1 = model(view1)
            feat2 = model(view2)
            
            # Move to CPU immediately so we don't run out of GPU memory over the dataset
            all_features_view1.append(feat1.cpu())
            all_features_view2.append(feat2.cpu())

    print("Concatenating features...")
    # Combine lists into massive tensors: [Dataset_Size, Feature_Dim]
    all_features_view1 = torch.cat(all_features_view1, dim=0)
    all_features_view2 = torch.cat(all_features_view2, dim=0)

    print(f"Feature shape extracted: {all_features_view1.shape}")

    # Save to disk
    torch.save({
        'view1': all_features_view1,
        'view2': all_features_view2
    }, save_path)
    
    print(f"Success! Features saved to: {save_path}")

if __name__ == "__main__":
    main()

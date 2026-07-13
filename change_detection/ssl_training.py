import sys,os
import argparse
import logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from simsiam.simsiam_loss import simsiam_loss
from feature_extractor import main as extract_features


class Projector(nn.Module):
 

    def __init__(self, in_dim):
        super().__init__(
            nn.Linear(in_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            )
    
    def forward(self, x):
        return self.net(x)
    
class SimSiamMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.projector = Projector(in_dim)
        self.predictor = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048)
        )

    def forward(self, f1, f2):
        # We start directly from features f1, f2!
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

class LatentMIM_MLP(nn.Module):
    """Masked Feature Modeling: Drops out random channels of the vector and reconstructs."""
    def __init__(self, in_dim, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.autoencoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, in_dim)
        )

    def forward(self, f1):
        # Randomly mask out channels
        mask = (torch.rand_like(f1) > self.mask_ratio).float()
        masked_f1 = f1 * mask
        
        # Reconstruct the original feature
        reconstructed_f1 = self.autoencoder(masked_f1)
        return reconstructed_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP Heads on Precomputed Features")
    parser.add_argument("--method", type=str, choices=["simsiam", "mim"], required=True)
    parser.add_argument("--batch_size", type=int, default=512, help="Can be massive now!")
    parser.add_argument("--extraction_bs", type=int, default=8, help="Batch size used during feature extraction")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset for feature extraction")
    parser.add_argument("--model_name", type=str, default="swin_tiny_patch4_window7_224", help="Backbone model name")
    parser.add_argument("--output_dir", type=str, default="features", help="Directory to save extracted features")
    return parser.parse_args()

def setup_logger(model_type, method, timestamp):
    """Sets up logging"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/pre_train_{model_type}_{method}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(args.model_name, args.method, timestamp=timestamp)

    features_path = f"features/features_{args.model_name.replace('/', '_')}.pt"
    if not os.path.exists(features_path):
        logger.info(f"Features not found at {features_path}. Extracting features...")
        extract_features(model_name=args.model_name, data_path=args.data_path, output_dir=args.output_dir, batch_size=args.extraction_bs) 
    else:
        logger.info(f"Features already exist at {features_path}. Skipping extraction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading precomputed features from {features_path}...")
    data = torch.load(features_path, map_location="cpu")
    v1_features, v2_features = data['view1'], data['view2']
    
    # Automatically determine feature dimension (768 for Swin, 384 for DINOv2)
    in_dim = v1_features.shape[1]
    logger.info(f"Loaded {len(v1_features)} samples with feature dimension {in_dim}.")

    # Create an ultra-fast Tensor DataLoader (No images, just vectors!)
    dataset = TensorDataset(v1_features, v2_features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    lr = args.base_lr  # Default learning rate; will be adjusted for SimSiam
    # Initialize requested SSL Method
    if args.method == "simsiam":
        model = SimSiamMLP(in_dim=in_dim).to(device)
        lr = args.base_lr * (args.batch_size / 256)
    elif args.method == "mim":
        model = LatentMIM_MLP(in_dim=in_dim).to(device)
        mim_criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    logger.info(f"Starting {args.method.upper()} Training for {args.epochs} Epochs...")
    
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for f1, f2 in pbar:
            f1, f2 = f1.to(device), f2.to(device)

            optimizer.zero_grad()
            
            if args.method == "simsiam":
                p1, p2, z1, z2 = model(f1, f2)
                loss = simsiam_loss(p1, p2, z1, z2)
            elif args.method == "mim":
                # For MIM, we just try to reconstruct f1 from a masked version of f1
                reconstructed = model(f1)
                loss = mim_criterion(reconstructed, f1)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(loader)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = f"checkpoints/best_{args.model_name}_{args.method}_mlp.pth"
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    logger.info(f"Training Complete! Best model saved to: {save_path}")

if __name__ == "__main__":
    main()
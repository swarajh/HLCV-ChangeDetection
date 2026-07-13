import sys
import os
import argparse
import logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from change_detection.dataset import LevirDataset
from change_detection.losses import DiceLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin Models")
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["mim", "simsiam", "sim","baseline"], 
        required=True,
        help="Specify which model variant to train."
    )
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default=None,
        help="Path to pretrained encoder weights (if any)."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
    )
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset.")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation dataset.")
    return parser.parse_args()

def setup_logger(model_type, timestamp):
    """Sets up logging"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/train_{model_type}_{timestamp}.log"
    
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
    
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Training dataset path does not exist: {args.train_path}")
    if not os.path.exists(args.val_path):
        raise FileNotFoundError(f"Validation dataset path does not exist: {args.val_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize logger
    logger = setup_logger(args.model_type, timestamp)
    logger.info(f"Script started. Training model: {args.model_type.upper()}")
    logger.info(f"Arguments: {vars(args)}")

    # Model selection based on argument
    if args.model_type == "mim":
        from change_detection.swin_model_mim import SwinChangeDetector
    elif args.model_type == "simsiam":
        from change_detection.swin_model_simsiam import SwinChangeDetector, SwinProjectorChangeDetector
    elif args.model_type == "sim":
        from change_detection.swin_model import SwinChangeDetector
    elif args.model_type == "baseline":
        from change_detection.baseline_model import SiameseBaseline
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    # Set Default encoder weights if not provided
    if args.encoder_weights is None:
        if args.model_type == "mim":
            args.encoder_weights = "checkpoints/mim_encoder.pth"
        elif args.model_type == "simsiam":
            args.encoder_weights = "checkpoints/simsiam_encoder_swin_tiny_patch4_window7_224_bs64.pth"
        logger.info(f"No encoder weights provided. Using default: {args.encoder_weights}")

    # Datasets
    logger.info("Creating datasets...")
    train_dataset = LevirDataset(args.train_path)
    val_dataset = LevirDataset(args.val_path)

    logger.info(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Device
    logger.info("Selecting device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Checkpoint Folder
    os.makedirs("checkpoints", exist_ok=True)
    logger.info("Checkpoint directory ready.")

    # Model
    logger.info(f"Creating {args.model_type} model...")
    save_path = f"checkpoints/swin_{args.model_type}_{timestamp}.pth"
    
    if args.model_type == "mim":
        model = SwinChangeDetector(pretrained=False, mim_weights=args.encoder_weights)
    elif args.model_type == "simsiam":
        if "mlp.pth" in args.encoder_weights:
            model = SwinProjectorChangeDetector(model_name=args.model_name,
                                                projector_weights=args.encoder_weights,
                                                freeze_backbone=True)
            save_path = f"checkpoints/swin_{args.model_type}_projector_{timestamp}.pth"
        else:
            model = SwinChangeDetector(pretrained=False, simsiam_weights=args.encoder_weights)
    elif args.model_type == "sim":
        model = SwinChangeDetector(pretrained=True)
    elif args.model_type == "baseline":
        model = SiameseBaseline()
    model = model.to(device)

    # Losses & Optimizer
    logger.info("Creating losses and optimizer...")
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-4)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training
    logger.info("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"--- Starting Epoch {epoch+1}/{args.epochs} ---")
        model.train()
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for image_a, image_b, label in pbar:

            image_a = image_a.to(device)
            image_b = image_b.to(device)
            label = label.to(device)

            output = model(image_a, image_b)
            loss = bce_loss(output, label) + dice_loss(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        running_val_loss = 0

        for image_a, image_b, label in val_loader:
            image_a = image_a.to(device)
            image_b = image_b.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = model(image_a, image_b)
                val_loss = bce_loss(output, label) + dice_loss(output, label)
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)

        logger.info(f"Epoch {epoch+1}/{args.epochs} Completed | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # =========================
        # Unified best-model saving logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path}! (Loss: {best_val_loss:.4f})")

        scheduler.step()

    # Saving the metrics to CSV to find the best model later 
    csv_folder = "csv_results"
    os.makedirs(csv_folder, exist_ok=True)
    csv_file = os.path.join(csv_folder, "experiment_results.csv")
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("Timestamp,Model_Type,Epochs,Batch_Size,Best_Val_Loss,Checkpoint_Path\n")
        
        # Write the data for this run
        f.write(f"{timestamp},{args.model_type},{args.epochs},{args.batch_size},{best_val_loss:.4f},{save_path}\n")

    logger.info("Training finished!")

if __name__ == "__main__":
    main()
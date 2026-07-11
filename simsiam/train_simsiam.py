import os
import sys
import argparse
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler,autocast

from simsiam.eurosat_dataset import EuroSATSimSiamDataset
from simsiam.simsiam_model import SimSiam
from simsiam.simsiam_loss import simsiam_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SimSiam Model")
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to pretraining dataset (e.g., datasets/EuroSAT)."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="swin_tiny_patch4_window7_224", 
        help="Name of the timm backbone model to use."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--base_lr", 
        type=float, 
        default=0.05, 
        help="Base learning rate for the SimSiam linear scaling rule."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0, 
        help="Number of workers for data loading."
    )
    parser.add_argument(
        "--to_save",
        type=bool,
        default=True,
        help="Whether to save the best model and encoder checkpoints."
    )
    return parser.parse_args()


def setup_logger(timestamp):
    """Sets up logging"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/pretrain_simsiam_{timestamp}.log"
    
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
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(timestamp)
    
    logger.info("SimSiam Pretraining script started.")
    logger.info(f"Arguments: {vars(args)}")

    # --------------------
    # Device Setup
    # --------------------
    logger.info("Selecting device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner for better performance
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # --------------------
    # Dataset & DataLoader
    # --------------------
    logger.info("Creating EuroSAT dataset...")
    dataset = EuroSATSimSiamDataset(args.data_path)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    logger.info(f"Dataset Size: {len(dataset)} | Number of batches: {len(loader)}")

    # --------------------
    # Model Setup
    # --------------------
    logger.info("Creating SimSiam model...")
    model = SimSiam(model_name=args.model_name)

    # If multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # --------------------
    # Optimizer Setup
    # --------------------
    # Baseline Setup from the SimSiam paper: https://arxiv.org/abs/2011.10566
    # Linear scaling rule dynamic calculation
    lr = args.base_lr * (args.batch_size / 256)
    logger.info(f"Calculated Learning Rate: {lr:.6f} (Base: {args.base_lr}, Batch Size: {args.batch_size})")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    # --------------------
    # Scheduler Setup
    # --------------------
    logger.info("Setting up CosineAnnealingLR scheduler...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,  # Total number of epochs
        eta_min=0           # Minimum learning rate
    )

    # --------------------
    # Checkpoint Setup
    # --------------------
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = f"checkpoints/simsiam_{args.model_name}_bs{args.batch_size}.pth"
    best_encoder_path = f"checkpoints/simsiam_encoder_{args.model_name}_bs{args.batch_size}.pth"
    logger.info("Checkpoint directory ready.")

    # --------------------
    # Training Loop
    # --------------------
    logger.info("Starting training...")
    best_loss = float("inf")
    scaler = GradScaler()  # For mixed precision training

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"--- Starting Epoch {epoch+1}/{args.epochs} | Learning Rate: {current_lr:.6f} ---")
        
        model.train()
        running_loss = 0

        # TQDM Progress Bar
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for view1, view2 in pbar:
            view1 = view1.to(device)
            view2 = view2.to(device)

            with autocast():
                p1, p2, z1, z2 = model(view1, view2)
                loss = simsiam_loss(p1, p2, z1, z2)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            # Update progress bar with the current batch loss
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        # Calculate epoch average loss
        avg_loss = running_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} Completed | Avg Loss: {avg_loss:.4f}")

        # Save best model logic
        if avg_loss < best_loss and args.to_save:
            best_loss = avg_loss
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            # Save the full SimSiam model
            torch.save(base_model.state_dict(), best_model_path)
            # Save JUST the encoder (for downstream tasks like Change Detection)
            torch.save(base_model.encoder.state_dict(), best_encoder_path)
            
            logger.info(f"New best model saved! (Loss: {best_loss:.4f})")
            logger.info(f"-> Full model: {best_model_path}")
            logger.info(f"-> Encoder only: {best_encoder_path}")

        # Step the learning rate scheduler
        scheduler.step()

    logger.info("SimSiam Pretraining finished!")

if __name__ == "__main__":
    main()
import sys
import os
import argparse
import csv
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from change_detection.dataset import LevirDataset
from change_detection.metrics import compute_iou, compute_f1

def get_best_checkpoint(model_type, csv_file="csv_results/experiment_results.csv"):
    """Reads the CSV file and returns the path to the best model for a specific type."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Cannot find {csv_file}. Have you trained any models yet?")

    best_loss = float('inf')
    best_ckpt = None

    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Model_Type"] == model_type:
                val_loss = float(row["Best_Val_Loss"])
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_ckpt = row["Checkpoint_Path"]
                    
    if best_ckpt is None:
        raise ValueError(f"No records found in {csv_file} for model_type: {model_type}")
        
    return best_ckpt, best_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Models")
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["mim", "simsiam", "sim", "baseline"], 
        required=True, 
        help="Model variant to evaluate."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to the saved model weights (.pth file). If omitted, fetches the best one from CSV."
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to test dataset."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
    )
    return parser.parse_args()

def setup_logger(model_type, timestamp):
    """Sets up logging"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/test_{model_type}_{timestamp}.log"
    
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
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(args.model_type, timestamp)

    logger.info(f"Testing script started for model: {args.model_type.upper()}")

    # Handle checkpoint fetching
    if args.checkpoint is None:
        logger.info(f"Searching for the best {args.model_type.upper()} model in CSV...")
        checkpoint_path, best_loss = get_best_checkpoint(args.model_type)
        logger.info("Found Best Model!")
        logger.info(f"Path: {checkpoint_path}")
        logger.info(f"Validation Loss: {best_loss:.4f}")
    else:
        checkpoint_path = args.checkpoint

    # 1. Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    image_size = 224  # Default image size for Swin models
    if "dinov2" in args.model_name:
        image_size = 518
    # 2. Dataset & DataLoader
    logger.info("Creating test dataset...")
    dataset = LevirDataset(args.data_path, image_size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Dataset size: {len(dataset)} | Number of batches: {len(loader)}")

    # 3. Dynamic Model Initialization
    logger.info("Creating model...")
    if args.model_type == "mim":
        from change_detection.swin_model_mim import SwinChangeDetector
        # We set mim_weights=None because we are loading a fully trained checkpoint
        model = SwinChangeDetector(pretrained=False, mim_weights=None)
    elif args.model_type == "simsiam":
        if "projector" in checkpoint_path.lower() and "swin" in checkpoint_path.lower():
            logger.info("Detected encoder checkpoint. Using SwinProjectorChangeDetector for evaluation.")
            from change_detection.swin_model_simsiam import SwinProjectorChangeDetector
            model = SwinProjectorChangeDetector(model_name=args.model_name,
                                                projector_weights=None,
                                                freeze_backbone=False)
        elif "projector" in checkpoint_path.lower() and "dino" in checkpoint_path.lower():
            logger.info("Detected encoder checkpoint. Using DINOProjectorChangeDetector for evaluation.")
            from change_detection.dino_model_simsiam import DINOProjectorChangeDetector
            model = DINOProjectorChangeDetector(model_name=args.model_name,
                                                projector_weights=None,
                                                freeze_backbone=False)
        else:
            from change_detection.swin_model_simsiam import SwinChangeDetector
            model = SwinChangeDetector(pretrained=False, simsiam_weights=None)
    elif args.model_type == "sim":
        from change_detection.swin_model import SwinChangeDetector
        model = SwinChangeDetector(pretrained=False)
    elif args.model_type == "baseline":
        from change_detection.baseline_model import SiameseBaseline
        model = SiameseBaseline()
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    # Load the trained weights
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return
    model = model.to(device)
    model.eval()

    # 4. Evaluation Loop
    logger.info("Starting evaluation...")
    total_iou = 0.0
    total_f1 = 0.0
    num_samples = 0

    with torch.no_grad():
        # Using tqdm to show a live updating progress bar with current scores
        pbar = tqdm(loader, desc="Testing", leave=True)
        
        for img_a, img_b, mask in pbar:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            mask = mask.to(device)

            prediction = model(img_a, img_b)
            
            # Safety check: interpolate prediction if spatial dimensions don't match the mask
            if prediction.shape != mask.shape:
                prediction = torch.nn.functional.interpolate(
                    prediction, size=mask.shape[2:], mode='bilinear', align_corners=False
                )

            # Convert raw logits to probabilities, then to binary (0 or 1)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > 0.5).float()

            # Accumulate metrics
            total_iou += compute_iou(prediction, mask)
            total_f1 += compute_f1(prediction, mask)
            num_samples += 1
            
            # Update the progress bar text with running averages
            pbar.set_postfix({
                "Avg IoU": f"{(total_iou / num_samples):.4f}", 
                "Avg F1": f"{(total_f1 / num_samples):.4f}"
            })

    # 5. Final Metrics Calculation
    avg_iou = total_iou / num_samples
    avg_f1 = total_f1 / num_samples

    csv_folder = "csv_results"
    os.makedirs(csv_folder, exist_ok=True)
    csv_file = os.path.join(csv_folder, "testing_results.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("Model_Type,Avg_IoU,Avg_F1,Checkpoint_Path\n")
        f.write(f"{args.model_type},{avg_iou:.4f},{avg_f1:.4f},{checkpoint_path}\n")
    logger.info(f"Results saved to {csv_file}")

    logger.info("="*40)
    logger.info(f"FINAL EVALUATION RESULTS: {args.model_type.upper()}")
    logger.info("="*40)
    logger.info(f"Validation IoU: {avg_iou:.4f}")
    logger.info(f"Validation F1 : {avg_f1:.4f}")
    logger.info("="*40)
    logger.info("Evaluation finished!")

if __name__ == "__main__":
    main()
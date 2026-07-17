import sys
import os
import argparse
import random
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from change_detection.dataset import LevirDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Change Detection Predictions")
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["mim", "simsiam", "sim", "baseline"], 
        required=True, 
        help="Model variant to visualize."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the saved model weights (.pth file) to evaluate."
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to the test dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model backbone name (e.g., vit_large_patch14_dinov2.lvd142m)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory where visualization plots will be saved."
    )
    return parser.parse_args()

def setup_logger(model_type, timestamp):
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/visualize_{model_type}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# def denormalize_image(tensor):
#     """Reverses DINOv2 / Swin normalization to render standard RGB imagery."""
#     # Assuming standard ImageNet/DINOv2 stats used in your LevirDataset
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
    
#     image = tensor.cpu().numpy().transpose(1, 2, 0)
#     image = (image * std) + mean
#     image = np.clip(image, 0, 1)
#     return image

def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(args.model_type, timestamp)
    
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    image_size = 224
    if "dinov2" in args.model_name:
        image_size = 518

    logger.info(f"Loading dataset from {args.data_path} (resizing to {image_size})...")
    dataset = LevirDataset(args.data_path, image_size=image_size)


    idx = random.randint(0, len(dataset) - 1)
    logger.info(f"Picking a random index to visualize: {idx}")

    img_a, img_b, mask = dataset[idx]

    logger.info(f"Recreating {args.model_type.upper()} architecture...")
    checkpoint_path = args.checkpoint
    
    if args.model_type == "mim":
        from change_detection.swin_model_mim import SwinChangeDetector
        model = SwinChangeDetector(pretrained=False, mim_weights=None)
    elif args.model_type == "simsiam":
        if "projector" in checkpoint_path.lower() and "swin" in checkpoint_path.lower():
            from change_detection.swin_model_simsiam import SwinProjectorChangeDetector
            model = SwinProjectorChangeDetector(model_name=args.model_name, projector_weights=None, freeze_backbone=False)
        elif "projector" in checkpoint_path.lower() and "dino" in checkpoint_path.lower():
            from change_detection.dino_model_simsiam import DINOProjectorChangeDetector
            model = DINOProjectorChangeDetector(model_name=args.model_name, projector_weights=None, freeze_backbone=False)
        elif "lora" in checkpoint_path.lower() and "dino" in checkpoint_path.lower():
            from change_detection.dino_model_simsiam import DINOLoRAProjectorChangeDetector
            model = DINOLoRAProjectorChangeDetector(model_name=args.model_name, projector_weights=None, freeze_backbone=False)
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

    logger.info(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()


    logger.info("Executing forward pass...")
    # Adding batch dimension [1, C, H, W] for inference
    batch_img_a = img_a.unsqueeze(0).to(device)
    batch_img_b = img_b.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(batch_img_a, batch_img_b)
        print("img_a      :", img_a.shape)
        print("img_b      :", img_b.shape)
        print("batch_img_a:", batch_img_a.shape)
        print("batch_img_b:", batch_img_b.shape)
        print("mask       :", mask.shape)
        print("prediction :", prediction.shape)
        
        if prediction.shape[-2:] != mask.shape[-2:]:
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        prediction = torch.sigmoid(prediction)
        pred_mask = (prediction > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

    gt_mask = mask.cpu().numpy()

    img_a_np = img_a.permute(1, 2, 0).cpu().numpy()
    img_b_np = img_b.permute(1, 2, 0).cpu().numpy()
    # Create the comparison figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(img_a_np)
    axes[0].set_title("Image A (T1)")
    axes[0].axis("off")
    
    axes[1].imshow(img_b_np)
    axes[1].set_title("Image B (T2)")
    axes[1].axis("off")
    
    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")
    
    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title("Predicted Mask")
    axes[3].axis("off")

    plt.tight_layout()
    
    save_filename = os.path.join(args.output_dir, f"vis_{args.model_name}_idx{idx}_{checkpoint_path}.png")
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Success! Visualization generated and saved to: {save_filename}")

if __name__ == "__main__":
    main()
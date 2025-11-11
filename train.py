import os
import yaml
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ccmim import CCMIM
from models.backbone.mim_block import MIMBlock
from models.fusion.ddf_module import DDFModule
from models.neck.spt_module import SPTModule
from utils.data_utils import ConcreteDefectDataset
from utils.metric_utils import calculate_metrics
from utils.log_utils import Logger
from utils.vis_utils import plot_loss_curve

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_optimizer(model, config):
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['train']['lr'],
        momentum=config['train']['momentum'],
        weight_decay=config['train']['weight_decay']
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['train']['lr_decay_steps'],
        gamma=0.1
    )
    return optimizer, lr_scheduler

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, config):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{config['train']['epochs']}]")
    
    for batch in pbar:
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        pbar.set_postfix({"Batch Loss": loss.item(), "Avg Loss": total_loss / len(train_loader.dataset)})
    
    avg_train_loss = total_loss / len(train_loader.dataset)
    return avg_train_loss

def validate(model, val_loader, criterion, metric_fn, device, config):
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate"):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_val_loss += loss.item() * images.size(0)
            all_preds.extend(torch.softmax(outputs['detections'], dim=-1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    metrics = metric_fn(
        all_preds, 
        all_targets, 
        iou_thres=config['test']['iou_thres'],
        conf_thres=config['test']['conf_thres']
    )
    return avg_val_loss, metrics

def main(config_path):
    # Initialize settings
    config = load_config(config_path)
    set_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directories
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    os.makedirs(config['train']['ckpt_dir'], exist_ok=True)
    
    # Initialize logger
    logger = Logger(os.path.join(config['train']['log_dir'], "train_log.txt"))
    logger.write(f"Training Configuration: {config}")
    
    # Load datasets
    train_dataset = ConcreteDefectDataset(
        data_dir=os.path.join(config['data']['root_dir'], config['data']['dataset'], "train"),
        image_size=config['data']['image_size'],
        augment=True
    )
    val_dataset = ConcreteDefectDataset(
        data_dir=os.path.join(config['data']['root_dir'], config['data']['dataset'], "val"),
        image_size=config['data']['image_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )
    
    # Initialize model, criterion, optimizer
    model = CCMIM(
        mim_hidden_dim=config['model']['mim']['hidden_dim'],
        ddf_channel=config['model']['ddf']['channel'],
        spt_token_level=config['model']['spt']['token_level'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    criterion = torch.nn.CombinedLoss(  # Custom combined loss for classification and localization
        cls_loss=torch.nn.CrossEntropyLoss(),
        reg_loss=torch.nn.IoULoss()
    )
    
    optimizer, lr_scheduler = get_optimizer(model, config)
    
    # Training loop
    best_val_map = 0.0
    train_losses = []
    val_losses = []
    val_maps = []
    
    for epoch in range(config['train']['epochs']):
        # Train
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, config)
        train_losses.append(avg_train_loss)
        
        # Validate
        avg_val_loss, val_metrics = validate(model, val_loader, criterion, calculate_metrics, device, config)
        val_losses.append(avg_val_loss)
        val_maps.append(val_metrics['mAP50'])
        
        # Update scheduler
        lr_scheduler.step()
        
        # Log results
        log_msg = (f"Epoch [{epoch+1}/{config['train']['epochs']}] | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"Precision: {val_metrics['precision']:.4f} | "
                   f"Recall: {val_metrics['recall']:.4f} | "
                   f"F1-Score: {val_metrics['f1']:.4f} | "
                   f"mAP50: {val_metrics['mAP50']:.4f}")
        logger.write(log_msg)
        print(log_msg)
        
        # Save best model
        if val_metrics['mAP50'] > best_val_map:
            best_val_map = val_metrics['mAP50']
            torch.save(model.state_dict(), os.path.join(config['train']['ckpt_dir'], "best_ccmim_model.pth"))
            logger.write(f"Best model saved (mAP50: {best_val_map:.4f})")
    
    # Save training results
    np.save(os.path.join(config['train']['log_dir'], "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(config['train']['log_dir'], "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(config['train']['log_dir'], "val_maps.npy"), np.array(val_maps))
    
    # Plot loss curve
    plot_loss_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=os.path.join(config['train']['log_dir'], "loss_curve.png")
    )
    
    logger.write(f"Training completed. Best val mAP50: {best_val_map:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CCMIM Model for Concrete Defect Detection")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to training config file")
    args = parser.parse_args()
    main(args.config)
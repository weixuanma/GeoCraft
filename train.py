import os
import yaml
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import hardware, dataset as dataset_config, train as train_config, log_save
from data.dataset_loader import ObjverseDataset
from data.data_augmentation import PointCloudAugmenter, ImageAugmenter
from models.diff2dpoint.diff2dpoint import Diff2DPoint
from models.point2dmesh.point2dmesh import Point2DMesh
from models.vision3dgen.vision3dgen import Vision3DGen
from trainers.base_trainer import BaseTrainer
from trainers.diff2dpoint_trainer import Diff2DPointTrainer
from trainers.point2dmesh_trainer import Point2DMeshTrainer
from trainers.vision3dgen_trainer import Vision3DGenTrainer
from utils.logger import Logger
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, save_config
from utils.visualization_utils import plot_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="GeoCraft 3D Reconstruction Training Script")
    parser.add_argument("--config_dir", type=str, default="./configs", help="Path to config directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--resume_epoch", type=int, default=-1, help="Resume from specific epoch (use -1 for last)")
    parser.add_argument("--stage", type=str, default="all", choices=["diff2dpoint", "point2dmesh", "vision3dgen", "all"], help="Training stage to run")
    return parser.parse_args()

def load_stage_configs(config_dir):
    with open(os.path.join(config_dir, "diff2dpoint_config.yaml"), "r") as f:
        diff2dpoint_config = yaml.safe_load(f)
    with open(os.path.join(config_dir, "point2dmesh_config.yaml"), "r") as f:
        point2dmesh_config = yaml.safe_load(f)
    with open(os.path.join(config_dir, "vision3dgen_config.yaml"), "r") as f:
        vision3dgen_config = yaml.safe_load(f)
    return diff2dpoint_config, point2dmesh_config, vision3dgen_config

def prepare_dataloader():
    point_aug = PointCloudAugmenter()
    img_aug = ImageAugmenter()
    transform = {
        "point_cloud": point_aug,
        "image": img_aug
    }
    
    train_dataset = ObjverseDataset(
        root_path=dataset_config["train"]["root_path"],
        sample_num=dataset_config["train"]["sample_num"],
        preprocess=dataset_config["train"]["preprocess"],
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hardware["batch_size"],
        shuffle=True,
        num_workers=hardware["num_workers"],
        pin_memory=True
    )
    return train_loader

def setup_device():
    device = torch.device(f"cuda:{hardware['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and len(hardware["gpu_ids"]) > 1:
        print(f"Using multiple GPUs: {hardware['gpu_ids']}")
    else:
        print(f"Using device: {device}")
    return device

def init_logger_and_configs(log_dir, config_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)
    
    save_config(hardware, log_dir, "hardware_config")
    save_config(dataset_config, log_dir, "dataset_config")
    save_config(train_config, log_dir, "train_config")
    
    diff2dpoint_config, point2dmesh_config, vision3dgen_config = load_stage_configs(config_dir)
    save_config(diff2dpoint_config, log_dir, "diff2dpoint_config")
    save_config(point2dmesh_config, log_dir, "point2dmesh_config")
    save_config(vision3dgen_config, log_dir, "vision3dgen_config")
    
    return logger, diff2dpoint_config, point2dmesh_config, vision3dgen_config

def train_diff2dpoint(train_loader, device, config, logger, checkpoint_dir, resume=False, resume_epoch=-1):
    model = Diff2DPoint(config).to(device)
    if len(hardware["gpu_ids"]) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=hardware["gpu_ids"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["optimizer"]["lr"],
        weight_decay=train_config["optimizer"]["weight_decay"],
        betas=train_config["optimizer"]["betas"]
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=1e-6
    )
    
    start_epoch = 0
    if resume:
        model, optimizer, start_epoch, _ = load_checkpoint(
            model, optimizer, os.path.join(checkpoint_dir, f"diff2dpoint_checkpoint_epoch_{resume_epoch:03d}.pth")
            if resume_epoch != -1 else checkpoint_dir
        )
        logger.info(f"Resumed Diff2DPoint training from epoch {start_epoch}")
    
    criterion = nn.MSELoss() if config["loss"]["type"] == "L2Loss" else nn.L1Loss()
    trainer = Diff2DPointTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        save_freq=log_save["save_freq"],
        eval_freq=log_save["eval_freq"]
    )
    
    metrics_history = trainer.train(
        train_loader=train_loader,
        epochs=config["train"]["epochs"],
        start_epoch=start_epoch,
        lr_scheduler=lr_scheduler
    )
    
    plot_metrics(metrics_history, log_save["log_dir"], title="Diff2DPoint Training Metrics")
    save_checkpoint(model, optimizer, config["train"]["epochs"], metrics_history["train_loss"][-1], checkpoint_dir, "diff2dpoint_final")
    logger.info("Diff2DPoint training completed")
    return model, metrics_history

def train_point2dmesh(train_loader, device, config, logger, checkpoint_dir, diff2dpoint_model, resume=False, resume_epoch=-1):
    model = Point2DMesh(config).to(device)
    if len(hardware["gpu_ids"]) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=hardware["gpu_ids"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["optimizer"]["lr"],
        weight_decay=train_config["optimizer"]["weight_decay"],
        betas=train_config["optimizer"]["betas"]
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=1e-6
    )
    
    start_epoch = 0
    if resume:
        model, optimizer, start_epoch, _ = load_checkpoint(
            model, optimizer, os.path.join(checkpoint_dir, f"point2dmesh_checkpoint_epoch_{resume_epoch:03d}.pth")
            if resume_epoch != -1 else checkpoint_dir
        )
        logger.info(f"Resumed Point2DMesh training from epoch {start_epoch}")
    
    trainer = Point2DMeshTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        save_freq=log_save["save_freq"],
        eval_freq=log_save["eval_freq"],
        dpo_temperature=config["dpo"]["temperature"]
    )
    
    metrics_history = trainer.train(
        train_loader=train_loader,
        epochs=config["train"]["epochs"],
        start_epoch=start_epoch,
        lr_scheduler=lr_scheduler,
        diff2dpoint_model=diff2dpoint_model
    )
    
    plot_metrics(metrics_history, log_save["log_dir"], title="Point2DMesh Training Metrics")
    save_checkpoint(model, optimizer, config["train"]["epochs"], metrics_history["train_loss"][-1], checkpoint_dir, "point2dmesh_final")
    logger.info("Point2DMesh training completed")
    return model, metrics_history

def train_vision3dgen(train_loader, device, config, logger, checkpoint_dir, diff2dpoint_model, point2dmesh_model, resume=False, resume_epoch=-1):
    model = Vision3DGen(config).to(device)
    if len(hardware["gpu_ids"]) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=hardware["gpu_ids"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["optimizer"]["lr"],
        weight_decay=train_config["optimizer"]["weight_decay"],
        betas=train_config["optimizer"]["betas"]
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=1e-6
    )
    
    start_epoch = 0
    if resume:
        model, optimizer, start_epoch, _ = load_checkpoint(
            model, optimizer, os.path.join(checkpoint_dir, f"vision3dgen_checkpoint_epoch_{resume_epoch:03d}.pth")
            if resume_epoch != -1 else checkpoint_dir
        )
        logger.info(f"Resumed Vision3DGen training from epoch {start_epoch}")
    
    criterion = nn.MSELoss()
    trainer = Vision3DGenTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        save_freq=log_save["save_freq"],
        eval_freq=log_save["eval_freq"]
    )
    
    metrics_history = trainer.train(
        train_loader=train_loader,
        epochs=config["train"]["epochs"],
        start_epoch=start_epoch,
        lr_scheduler=lr_scheduler,
        diff2dpoint_model=diff2dpoint_model,
        point2dmesh_model=point2dmesh_model
    )
    
    plot_metrics(metrics_history, log_save["log_dir"], title="Vision3DGen Training Metrics")
    save_checkpoint(model, optimizer, config["train"]["epochs"], metrics_history["train_loss"][-1], checkpoint_dir, "vision3dgen_final")
    logger.info("Vision3DGen training completed")
    return model, metrics_history

def main():
    args = parse_args()
    
    device = setup_device()
    logger, diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg = init_logger_and_configs(log_save["log_dir"], args.config_dir)
    train_loader = prepare_dataloader()
    
    checkpoint_dir = os.path.join(log_save["checkpoint_dir"], "geocraft_train")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    diff2dpoint_model = None
    point2dmesh_model = None
    
    if args.stage in ["diff2dpoint", "all"]:
        logger.info("="*50)
        logger.info("Starting Diff2DPoint Stage Training")
        logger.info("="*50)
        diff2dpoint_model, _ = train_diff2dpoint(
            train_loader=train_loader,
            device=device,
            config=diff2dpoint_cfg,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
            resume_epoch=args.resume_epoch
        )
    
    if args.stage in ["point2dmesh", "all"]:
        if diff2dpoint_model is None and not args.resume:
            logger.error("Diff2DPoint model not found. Train Diff2DPoint first or enable resume.")
            return
        
        logger.info("="*50)
        logger.info("Starting Point2DMesh Stage Training")
        logger.info("="*50)
        point2dmesh_model, _ = train_point2dmesh(
            train_loader=train_loader,
            device=device,
            config=point2dmesh_cfg,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            diff2dpoint_model=diff2dpoint_model,
            resume=args.resume,
            resume_epoch=args.resume_epoch
        )
    
    if args.stage in ["vision3dgen", "all"]:
        if diff2dpoint_model is None or point2dmesh_model is None and not args.resume:
            logger.error("Diff2DPoint/Point2DMesh models not found. Train them first or enable resume.")
            return
        
        logger.info("="*50)
        logger.info("Starting Vision3DGen Stage Training")
        logger.info("="*50)
        _, _ = train_vision3dgen(
            train_loader=train_loader,
            device=device,
            config=vision3dgen_cfg,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            diff2dpoint_model=diff2dpoint_model,
            point2dmesh_model=point2dmesh_model,
            resume=args.resume,
            resume_epoch=args.resume_epoch
        )
    
    logger.info("="*50)
    logger.info("All Selected Training Stages Completed Successfully")
    logger.info("="*50)

if __name__ == "__main__":
    main()
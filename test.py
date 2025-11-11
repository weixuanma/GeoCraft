import os
import yaml
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from configs.base_config import hardware, dataset as dataset_config, log_save
from data.dataset_loader import GSODataset, Pix3DDataset
from models.diff2dpoint.diff2dpoint import Diff2DPoint
from models.point2dmesh.point2dmesh import Point2DMesh
from models.vision3dgen.vision3dgen import Vision3DGen
from testers.base_tester import BaseTester
from testers.metric_calculator import MetricCalculator
from testers.result_analyzer import ResultAnalyzer
from utils.logger import Logger
from utils.checkpoint_utils import load_checkpoint, load_config
from utils.visualization_utils import save_3d_surface, compare_results, plot_metrics
from utils.geometry_utils import compute_sdf

def parse_args():
    parser = argparse.ArgumentParser(description="GeoCraft 3D Reconstruction Testing Script")
    parser.add_argument("--config_dir", type=str, default="./configs", help="Path to config directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to trained model checkpoints")
    parser.add_argument("--dataset", type=str, required=True, choices=["gso", "pix3d"], help="Test dataset (GSO/Pix3D)")
    parser.add_argument("--save_results", action="store_true", help="Save 3D reconstruction results (point cloud/mesh/surface)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of results (comparison plots/metrics)")
    parser.add_argument("--eval_metrics", nargs="+", default=["CMMD", "FID_CLIP", "CLIP-score", "LPIPS"], help="Metrics to evaluate")
    return parser.parse_args()

def load_stage_configs(config_dir):
    diff2dpoint_cfg = load_config(os.path.join(config_dir, "diff2dpoint_config.yaml"))
    point2dmesh_cfg = load_config(os.path.join(config_dir, "point2dmesh_config.yaml"))
    vision3dgen_cfg = load_config(os.path.join(config_dir, "vision3dgen_config.yaml"))
    base_cfg = load_config(os.path.join(config_dir, "base_config.yaml"))
    return diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg, base_cfg

def prepare_test_dataloader(dataset_name, dataset_config):
    transform = {
        "point_cloud": lambda x: x,
        "image": lambda x: x
    }
    
    if dataset_name == "gso":
        test_dataset = GSODataset(
            root_path=dataset_config["test_gso"]["root_path"],
            sample_num=dataset_config["test_gso"]["sample_num"],
            transform=transform
        )
    elif dataset_name == "pix3d":
        test_dataset = Pix3DDataset(
            root_path=dataset_config["test_pix3d"]["root_path"],
            sample_num=dataset_config["test_pix3d"]["sample_num"],
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=hardware["batch_size"],
        shuffle=False,
        num_workers=hardware["num_workers"],
        pin_memory=True
    )
    return test_loader, test_dataset

def setup_device():
    device = torch.device(f"cuda:{hardware['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device for testing: {device}")
    return device

def load_trained_models(checkpoint_dir, diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg, device):
    # Load Diff2DPoint model
    diff2dpoint_model = Diff2DPoint(diff2dpoint_cfg).to(device)
    diff2dpoint_ckpt = os.path.join(checkpoint_dir, "diff2dpoint_final.pth")
    diff2dpoint_model, _, _, _ = load_checkpoint(diff2dpoint_model, None, diff2dpoint_ckpt)
    diff2dpoint_model.eval()
    
    # Load Point2DMesh model
    point2dmesh_model = Point2DMesh(point2dmesh_cfg).to(device)
    point2dmesh_ckpt = os.path.join(checkpoint_dir, "point2dmesh_final.pth")
    point2dmesh_model, _, _, _ = load_checkpoint(point2dmesh_model, None, point2dmesh_ckpt)
    point2dmesh_model.eval()
    
    # Load Vision3DGen model
    vision3dgen_model = Vision3DGen(vision3dgen_cfg).to(device)
    vision3dgen_ckpt = os.path.join(checkpoint_dir, "vision3dgen_final.pth")
    vision3dgen_model, _, _, _ = load_checkpoint(vision3dgen_model, None, vision3dgen_ckpt)
    vision3dgen_model.eval()
    
    # Use DataParallel if multiple GPUs
    if len(hardware["gpu_ids"]) > 1 and torch.cuda.is_available():
        diff2dpoint_model = nn.DataParallel(diff2dpoint_model, device_ids=hardware["gpu_ids"])
        point2dmesh_model = nn.DataParallel(point2dmesh_model, device_ids=hardware["gpu_ids"])
        vision3dgen_model = nn.DataParallel(vision3dgen_model, device_ids=hardware["gpu_ids"])
    
    return diff2dpoint_model, point2dmesh_model, vision3dgen_model

def init_test_logger_and_dirs(log_dir, dataset_name):
    test_log_dir = os.path.join(log_dir, f"test_{dataset_name}")
    test_result_dir = os.path.join(test_log_dir, "results")
    test_vis_dir = os.path.join(test_log_dir, "visualizations")
    
    os.makedirs(test_log_dir, exist_ok=True)
    os.makedirs(test_result_dir, exist_ok=True)
    os.makedirs(test_vis_dir, exist_ok=True)
    
    logger = Logger(test_log_dir, log_name=f"test_{dataset_name}")
    return logger, test_result_dir, test_vis_dir

def run_geocraft_inference(batch, diff2dpoint_model, point2dmesh_model, vision3dgen_model, device):
    img = batch["image"].to(device)
    gt_pc = batch["point_cloud"].to(device)
    batch_size = img.shape[0]
    
    # Dummy camera parameters (use precomputed if available)
    K = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    Rt = torch.eye(4, device=device)[:3, :4].unsqueeze(0).expand(batch_size, -1, -1)
    
    # Stage 1: Generate point cloud with Diff2DPoint
    with torch.no_grad():
        pred_pc = diff2dpoint_model.generate_point_cloud(img, K, Rt, num_points=gt_pc.shape[1])
    
    # Stage 2: Generate mesh with Point2DMesh
    with torch.no_grad():
        pred_meshes = point2dmesh_model.generate_mesh(pred_pc, post_process=True)
    
    # Stage 3: Generate 3D surface with Vision3DGen
    with torch.no_grad():
        pred_surfaces = [
            vision3dgen_model.generate_3d_surface(
                img[i].unsqueeze(0), 
                pred_pc[i].unsqueeze(0), 
                [pred_meshes[i]]
            ) for i in range(batch_size)
        ]
    
    # Prepare GT mesh (sample from GT point cloud for comparison)
    gt_meshes = []
    for pc in gt_pc:
        gt_mesh = point2dmesh_model.mesh_generator.generate_mesh(
            pc, 
            torch.tensor([[0,1,2]], dtype=torch.int64, device=device)
        )
        gt_meshes.append(gt_mesh)
    
    return {
        "gt_pc": gt_pc,
        "pred_pc": pred_pc,
        "gt_mesh": gt_meshes,
        "pred_mesh": pred_meshes,
        "pred_surface": pred_surfaces,
        "img": img,
        "paths": batch["path"]
    }

def evaluate_metrics(results, metric_calculator, eval_metrics):
    metrics = {}
    gt_pc = results["gt_pc"]
    pred_pc = results["pred_pc"]
    gt_mesh = results["gt_mesh"]
    pred_mesh = results["pred_mesh"]
    img = results["img"]
    
    if "CMMD" in eval_metrics:
        metrics["CMMD"] = metric_calculator.calculate_cmmd(gt_pc, pred_pc)
    if "FID_CLIP" in eval_metrics:
        metrics["FID_CLIP"] = metric_calculator.calculate_fid_clip(gt_mesh, pred_mesh, img)
    if "CLIP-score" in eval_metrics:
        metrics["CLIP-score"] = metric_calculator.calculate_clip_score(pred_mesh, img)
    if "LPIPS" in eval_metrics:
        metrics["LPIPS"] = metric_calculator.calculate_lpips(gt_mesh, pred_mesh, img)
    
    return metrics

def save_test_results(results, save_dir, batch_idx):
    batch_size = len(results["paths"])
    for i in range(batch_size):
        sample_name = os.path.basename(results["paths"][i]).split(".")[0]
        sample_dir = os.path.join(save_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save point cloud (GT + Pred)
        gt_pc_np = results["gt_pc"][i].cpu().detach().numpy()
        pred_pc_np = results["pred_pc"][i].cpu().detach().numpy()
        np.save(os.path.join(sample_dir, "gt_point_cloud.npy"), gt_pc_np)
        np.save(os.path.join(sample_dir, "pred_point_cloud.npy"), pred_pc_np)
        
        # Save mesh (GT + Pred)
        gt_mesh = results["gt_mesh"][i]
        pred_mesh = results["pred_mesh"][i]
        pred_surface = results["pred_surface"][i]
        import open3d as o3d
        o3d.io.write_triangle_mesh(os.path.join(sample_dir, "gt_mesh.obj"), gt_mesh)
        o3d.io.write_triangle_mesh(os.path.join(sample_dir, "pred_mesh.obj"), pred_mesh)
        o3d.io.write_triangle_mesh(os.path.join(sample_dir, "pred_surface.ply"), pred_surface)

def main():
    args = parse_args()
    
    # Load configs
    diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg, base_cfg = load_stage_configs(args.config_dir)
    
    # Setup device, dataloader, logger
    device = setup_device()
    test_loader, test_dataset = prepare_test_dataloader(args.dataset, dataset_config)
    logger, test_result_dir, test_vis_dir = init_test_logger_and_dirs(log_save["log_dir"], args.dataset)
    
    # Load trained models
    logger.info("Loading trained GeoCraft models...")
    diff2dpoint_model, point2dmesh_model, vision3dgen_model = load_trained_models(
        args.checkpoint_dir, diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg, device
    )
    logger.info("Models loaded successfully")
    
    # Init tester components
    metric_calculator = MetricCalculator(device=device)
    result_analyzer = ResultAnalyzer()
    tester = BaseTester(
        device=device,
        logger=logger,
        metric_calculator=metric_calculator,
        result_analyzer=result_analyzer
    )
    
    # Run testing
    logger.info(f"Starting testing on {args.dataset.upper()} dataset (Total samples: {len(test_dataset)})")
    logger.info(f"Evaluating metrics: {', '.join(args.eval_metrics)}")
    
    all_metrics = {metric: [] for metric in args.eval_metrics}
    total_batches = len(test_loader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            # Run end-to-end inference
            results = run_geocraft_inference(
                batch, diff2dpoint_model, point2dmesh_model, vision3dgen_model, device
            )
            
            # Evaluate metrics for current batch
            batch_metrics = evaluate_metrics(results, metric_calculator, args.eval_metrics)
            for metric, val in batch_metrics.items():
                all_metrics[metric].append(val)
            
            # Log batch metrics
            batch_metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in batch_metrics.items()])
            logger.info(f"Batch {batch_idx + 1} Metrics: {batch_metric_str}")
            
            # Save results if enabled
            if args.save_results:
                save_test_results(results, test_result_dir, batch_idx)
                logger.info(f"Batch {batch_idx + 1} results saved to {test_result_dir}")
            
            # Visualize if enabled
            if args.visualize and batch_idx % 5 == 0:  # Visualize every 5 batches
                sample_idx = 0  # Visualize first sample in batch
                sample_name = os.path.basename(results["paths"][sample_idx]).split(".")[0]
                vis_sample_dir = os.path.join(test_vis_dir, sample_name)
                os.makedirs(vis_sample_dir, exist_ok=True)
                
                # Generate comparison plot (GT vs Pred)
                compare_results(
                    gt_pc=results["gt_pc"][sample_idx],
                    pred_pc=results["pred_pc"][sample_idx],
                    gt_mesh=results["gt_mesh"][sample_idx],
                    pred_mesh=results["pred_mesh"][sample_idx],
                    save_path=vis_sample_dir
                )
                
                # Save surface rendering
                save_3d_surface(
                    surface_mesh=results["pred_surface"][sample_idx],
                    save_path=vis_sample_dir,
                    format="ply"
                )
                logger.info(f"Batch {batch_idx + 1} visualization saved to {vis_sample_dir}")
    
    # Calculate and log overall metrics
    logger.info("="*60)
    logger.info(f"Overall Test Metrics on {args.dataset.upper()} Dataset")
    logger.info("="*60)
    overall_metrics = {}
    for metric, vals in all_metrics.items():
        overall_val = np.mean(vals)
        overall_metrics[metric] = overall_val
        logger.info(f"{metric}: {overall_val:.6f}")
    
    # Save overall metrics and plot
    np.save(os.path.join(test_log_dir, "overall_metrics.npy"), overall_metrics)
    if args.visualize:
        plot_metrics(
            metrics_dict={k: [v] for k, v in overall_metrics.items()},
            save_path=test_vis_dir,
            title=f"GeoCraft Overall Metrics ({args.dataset.upper()})"
        )
        logger.info(f"Overall metrics plot saved to {test_vis_dir}")
    
    logger.info("="*60)
    logger.info("Testing Completed Successfully")
    logger.info("="*60)

if __name__ == "__main__":
    main()
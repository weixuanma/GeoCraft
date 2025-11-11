import os
import yaml
import torch
import argparse
import open3d as o3d
from PIL import Image
import numpy as np
from torchvision import transforms

from configs.base_config import hardware
from models.diff2dpoint.diff2dpoint import Diff2DPoint
from models.point2dmesh.point2dmesh import Point2DMesh
from models.vision3dgen.vision3dgen import Vision3DGen
from utils.checkpoint_utils import load_checkpoint, load_config
from utils.visualization_utils import render_point_cloud, render_mesh, save_3d_surface
from utils.geometry_utils import get_camera_projection_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="GeoCraft End-to-End 3D Reconstruction Inference Script")
    parser.add_argument("--input_img", type=str, required=True, help="Path to input RGB image (e.g., ./input.jpg)")
    parser.add_argument("--config_dir", type=str, default="./configs", help="Path to config directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to trained model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Path to save inference results")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points in generated point cloud")
    parser.add_argument("--mesh_post_process", action="store_true", help="Apply post-processing (smoothing) to generated mesh")
    parser.add_argument("--surface_resolution", type=int, default=256, help="Resolution for 3D surface extraction (Marching Cubes)")
    parser.add_argument("--camera_intrinsics", type=list, default=[500, 500, 256, 256], help="Camera intrinsics [fx, fy, cx, cy]")
    parser.add_argument("--camera_pose", type=list, default=[1,0,0,0, 0,1,0,0, 0,0,1,2], help="Camera extrinsic (3x4 matrix, flattened)")
    return parser.parse_args()

def load_configs(config_dir):
    """Load configs for all three GeoCraft stages"""
    diff2dpoint_cfg = load_config(os.path.join(config_dir, "diff2dpoint_config.yaml"))
    point2dmesh_cfg = load_config(os.path.join(config_dir, "point2dmesh_config.yaml"))
    vision3dgen_cfg = load_config(os.path.join(config_dir, "vision3dgen_config.yaml"))
    return diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg

def setup_device():
    """Setup computation device (GPU/CPU)"""
    device = torch.device(f"cuda:{hardware['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")
    return device

def load_trained_models(checkpoint_dir, diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg, device):
    """Load pre-trained GeoCraft models (Diff2DPoint → Point2DMesh → Vision3DGen)"""
    # Load Diff2DPoint
    diff2dpoint_model = Diff2DPoint(diff2dpoint_cfg).to(device)
    diff2dpoint_ckpt = os.path.join(checkpoint_dir, "diff2dpoint_final.pth")
    if not os.path.exists(diff2dpoint_ckpt):
        raise FileNotFoundError(f"Diff2DPoint checkpoint not found: {diff2dpoint_ckpt}")
    diff2dpoint_model, _, _, _ = load_checkpoint(diff2dpoint_model, None, diff2dpoint_ckpt)
    diff2dpoint_model.eval()

    # Load Point2DMesh
    point2dmesh_model = Point2DMesh(point2dmesh_cfg).to(device)
    point2dmesh_ckpt = os.path.join(checkpoint_dir, "point2dmesh_final.pth")
    if not os.path.exists(point2dmesh_ckpt):
        raise FileNotFoundError(f"Point2DMesh checkpoint not found: {point2dmesh_ckpt}")
    point2dmesh_model, _, _, _ = load_checkpoint(point2dmesh_model, None, point2dmesh_ckpt)
    point2dmesh_model.eval()

    # Load Vision3DGen
    vision3dgen_model = Vision3DGen(vision3dgen_cfg).to(device)
    vision3dgen_ckpt = os.path.join(checkpoint_dir, "vision3dgen_final.pth")
    if not os.path.exists(vision3dgen_ckpt):
        raise FileNotFoundError(f"Vision3DGen checkpoint not found: {vision3dgen_ckpt}")
    vision3dgen_model, _, _, _ = load_checkpoint(vision3dgen_model, None, vision3dgen_ckpt)
    vision3dgen_model.eval()

    # Multi-GPU support (if available)
    if len(hardware["gpu_ids"]) > 1 and torch.cuda.is_available():
        diff2dpoint_model = torch.nn.DataParallel(diff2dpoint_model, device_ids=hardware["gpu_ids"])
        point2dmesh_model = torch.nn.DataParallel(point2dmesh_model, device_ids=hardware["gpu_ids"])
        vision3dgen_model = torch.nn.DataParallel(vision3dgen_model, device_ids=hardware["gpu_ids"])
    
    return diff2dpoint_model, point2dmesh_model, vision3dgen_model

def preprocess_input_image(input_img_path, device):
    """Preprocess input RGB image (resize, normalize, tensor conversion)"""
    # Image transform (matches training preprocessing)
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    img = Image.open(input_img_path).convert("RGB")
    img_tensor = img_transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return img_tensor, img

def prepare_camera_parameters(intrinsics_list, pose_list, device):
    """Convert camera intrinsics/extrinsics to tensor matrices"""
    batch_size = 1

    # Intrinsics (K: 3x3)
    fx, fy, cx, cy = intrinsics_list
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device=device).unsqueeze(0).expand(batch_size, -1, -1)

    # Extrinsics (Rt: 3x4)
    Rt = torch.tensor(pose_list, device=device).reshape(3, 4).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Projection matrix (P = K @ Rt)
    P = get_camera_projection_matrix(K, Rt)
    return K, Rt, P

def init_output_dir(output_dir):
    """Create output directories for different result types"""
    output_subdirs = {
        "point_cloud": os.path.join(output_dir, "point_cloud"),
        "mesh": os.path.join(output_dir, "mesh"),
        "3d_surface": os.path.join(output_dir, "3d_surface"),
        "visualizations": os.path.join(output_dir, "visualizations")
    }
    for dir_path in output_subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return output_subdirs

def run_diff2dpoint_inference(model, img_tensor, K, Rt, num_points, device):
    """Stage 1: Generate geometrically aligned point cloud from input image"""
    with torch.no_grad():
        pred_pc = model.generate_point_cloud(
            img=img_tensor,
            K=K,
            Rt=Rt,
            num_points=num_points
        )
    # Convert to numpy for saving/visualization
    pred_pc_np = pred_pc.squeeze(0).cpu().detach().numpy()
    return pred_pc, pred_pc_np

def run_point2dmesh_inference(model, pred_pc, post_process, device):
    """Stage 2: Convert point cloud to high-quality mesh"""
    with torch.no_grad():
        pred_meshes = model.generate_mesh(
            point_cloud=pred_pc,
            post_process=post_process
        )
    # Extract single mesh (remove batch dimension)
    pred_mesh = pred_meshes[0] if isinstance(pred_meshes, list) else pred_meshes.squeeze(0)
    return pred_mesh

def run_vision3dgen_inference(model, img_tensor, pred_pc, pred_mesh, surface_resolution, device):
    """Stage 3: Generate high-fidelity 3D surface from multimodal features"""
    # Adjust input shapes for single-sample inference
    img_tensor = img_tensor.squeeze(0).unsqueeze(0)  # Ensure [1, 3, H, W]
    pred_pc = pred_pc.squeeze(0).unsqueeze(0)        # Ensure [1, N, 3]

    with torch.no_grad():
        pred_surface = model.generate_3d_surface(
            img=img_tensor,
            point_cloud=pred_pc,
            mesh=[pred_mesh]  # Wrap in list for batch compatibility
        )
    return pred_surface

def save_inference_results(pred_pc_np, pred_mesh, pred_surface, output_subdirs):
    """Save raw inference results (point cloud, mesh, 3D surface)"""
    # Save point cloud (.ply)
    pc_save_path = os.path.join(output_subdirs["point_cloud"], "pred_point_cloud.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pc_np)
    o3d.io.write_point_cloud(pc_save_path, pcd)

    # Save mesh (.obj)
    mesh_save_path = os.path.join(output_subdirs["mesh"], "pred_mesh.obj")
    o3d.io.write_triangle_mesh(mesh_save_path, pred_mesh)

    # Save 3D surface (.ply)
    surface_save_path, _ = save_3d_surface(
        surface_mesh=pred_surface,
        save_path=output_subdirs["3d_surface"],
        format="ply"
    )

    return pc_save_path, mesh_save_path, surface_save_path

def generate_visualizations(img, pred_pc_np, pred_mesh, pred_surface, output_subdirs):
    """Generate 2D visualizations of inference results"""
    # Save original input image
    img_save_path = os.path.join(output_subdirs["visualizations"], "input_image.png")
    img.save(img_save_path)

    # Render point cloud
    pc_render_path = os.path.join(output_subdirs["visualizations"], "point_cloud_render.png")
    pc_render = render_point_cloud(pred_pc_np, output_subdirs["visualizations"], color=[0.2, 0.6, 0.9])
    pc_render.save(pc_render_path)

    # Render mesh
    mesh_render_path = os.path.join(output_subdirs["visualizations"], "mesh_render.png")
    mesh_render = render_mesh(pred_mesh, output_subdirs["visualizations"], color=[0.8, 0.4, 0.2])
    mesh_render.save(mesh_render_path)

    # Render 3D surface
    surface_render_path = os.path.join(output_subdirs["visualizations"], "3d_surface_render.png")
    surface_render, _ = save_3d_surface(
        surface_mesh=pred_surface,
        save_path=output_subdirs["visualizations"],
        format="png"
    )
    surface_render.save(surface_render_path)

    return img_save_path, pc_render_path, mesh_render_path, surface_render_path

def main():
    args = parse_args()

    # 1. Load configs and setup environment
    print("Loading configs...")
    diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg = load_configs(args.config_dir)
    
    print("Setting up device...")
    device = setup_device()

    # 2. Load pre-trained models
    print("Loading trained GeoCraft models...")
    diff2dpoint_model, point2dmesh_model, vision3dgen_model = load_trained_models(
        args.checkpoint_dir, diff2dpoint_cfg, point2dmesh_cfg, vision3dgen_cfg, device
    )
    print("Models loaded successfully")

    # 3. Preprocess input
    print(f"Preprocessing input image: {args.input_img}")
    img_tensor, raw_img = preprocess_input_image(args.input_img, device)
    
    print("Preparing camera parameters...")
    K, Rt, _ = prepare_camera_parameters(args.camera_intrinsics, args.camera_pose, device)

    # 4. Init output directories
    print(f"Initializing output directory: {args.output_dir}")
    output_subdirs = init_output_dir(args.output_dir)

    # 5. Run end-to-end inference
    print("\n=== Starting GeoCraft Inference ===")
    
    # Stage 1: Diff2DPoint (Image → Point Cloud)
    print("Running Diff2DPoint (Stage 1/3)...")
    pred_pc, pred_pc_np = run_diff2dpoint_inference(
        model=diff2dpoint_model,
        img_tensor=img_tensor,
        K=K,
        Rt=Rt,
        num_points=args.num_points,
        device=device
    )
    print(f"Generated point cloud with {args.num_points} points")

    # Stage 2: Point2DMesh (Point Cloud → Mesh)
    print("Running Point2DMesh (Stage 2/3)...")
    pred_mesh = run_point2dmesh_inference(
        model=point2dmesh_model,
        pred_pc=pred_pc,
        post_process=args.mesh_post_process,
        device=device
    )
    print(f"Generated mesh with {len(pred_mesh.vertices)} vertices and {len(pred_mesh.triangles)} triangles")

    # Stage 3: Vision3DGen (Mesh → High-Fidelity 3D Surface)
    print("Running Vision3DGen (Stage 3/3)...")
    pred_surface = run_vision3dgen_inference(
        model=vision3dgen_model,
        img_tensor=img_tensor,
        pred_pc=pred_pc,
        pred_mesh=pred_mesh,
        surface_resolution=args.surface_resolution,
        device=device
    )
    print(f"Generated 3D surface with resolution {args.surface_resolution}x{args.surface_resolution}")

    # 6. Save results and visualizations
    print("\n=== Saving Results ===")
    # Save raw 3D assets
    pc_save, mesh_save, surface_save = save_inference_results(
        pred_pc_np, pred_mesh, pred_surface, output_subdirs
    )
    print(f"Point cloud saved to: {pc_save}")
    print(f"Mesh saved to: {mesh_save}")
    print(f"3D surface saved to: {surface_save}")

    # Generate and save 2D visualizations
    img_vis, pc_vis, mesh_vis, surface_vis = generate_visualizations(
        raw_img, pred_pc_np, pred_mesh, pred_surface, output_subdirs
    )
    print(f"Input image visualization saved to: {img_vis}")
    print(f"Point cloud render saved to: {pc_vis}")
    print(f"Mesh render saved to: {mesh_vis}")
    print(f"3D surface render saved to: {surface_vis}")

    # 7. Completion message
    print("\n=== Inference Completed Successfully ===")
    print(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
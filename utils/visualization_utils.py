import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

def render_point_cloud(pc, save_path, color=[0.5, 0.5, 0.5]):
    if isinstance(pc, torch.Tensor):
        pc_np = pc.cpu().detach().numpy()
    else:
        pc_np = pc
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (pc_np.shape[0], 1)))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    
    img_path = os.path.join(save_path, "point_cloud.png")
    vis.capture_screen_image(img_path)
    vis.destroy_window()
    
    img = Image.open(img_path)
    return img

def render_mesh(mesh, save_path, color=[0.8, 0.6, 0.4]):
    if isinstance(mesh, list):
        mesh = mesh[0]
    
    mesh_vis = o3d.geometry.TriangleMesh()
    mesh_vis.vertices = mesh.vertices
    mesh_vis.triangles = mesh.triangles
    mesh_vis.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(mesh.vertices), 1)))
    mesh_vis.compute_vertex_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh_vis)
    vis.get_render_option().mesh_show_back_face = True
    
    img_path = os.path.join(save_path, "mesh.png")
    vis.capture_screen_image(img_path)
    vis.destroy_window()
    
    img = Image.open(img_path)
    return img

def plot_metrics(metrics_dict, save_path, title="Training Metrics"):
    plt.figure(figsize=(12, 6))
    
    epochs = list(range(1, len(next(iter(metrics_dict.values()))) + 1))
    for metric_name, metric_values in metrics_dict.items():
        plt.plot(epochs, metric_values, label=metric_name, linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    metric_path = os.path.join(save_path, "metrics_plot.png")
    plt.savefig(metric_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    img = Image.open(metric_path)
    return img

def save_3d_surface(surface_mesh, save_path, format="ply"):
    os.makedirs(save_path, exist_ok=True)
    
    if format == "ply":
        mesh_path = os.path.join(save_path, "3d_surface.ply")
        o3d.io.write_triangle_mesh(mesh_path, surface_mesh)
    elif format == "obj":
        mesh_path = os.path.join(save_path, "3d_surface.obj")
        o3d.io.write_triangle_mesh(mesh_path, surface_mesh)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'ply' or 'obj'.")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(surface_mesh)
    vis.get_render_option().mesh_show_back_face = True
    
    img_path = os.path.join(save_path, "3d_surface_render.png")
    vis.capture_screen_image(img_path)
    vis.destroy_window()
    
    img = Image.open(img_path)
    return img, mesh_path

def compare_results(gt_pc, pred_pc, gt_mesh, pred_mesh, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    gt_pc_img = render_point_cloud(gt_pc, os.path.join(save_path, "gt_point_cloud"), color=[0.2, 0.8, 0.2])
    pred_pc_img = render_point_cloud(pred_pc, os.path.join(save_path, "pred_point_cloud"), color=[0.8, 0.2, 0.2])
    
    gt_mesh_img = render_mesh(gt_mesh, os.path.join(save_path, "gt_mesh"), color=[0.2, 0.2, 0.8])
    pred_mesh_img = render_mesh(pred_mesh, os.path.join(save_path, "pred_mesh"), color=[0.8, 0.8, 0.2])
    
    combined_img = Image.new("RGB", (gt_pc_img.width * 2, gt_pc_img.height * 2))
    combined_img.paste(gt_pc_img, (0, 0))
    combined_img.paste(pred_pc_img, (gt_pc_img.width, 0))
    combined_img.paste(gt_mesh_img, (0, gt_pc_img.height))
    combined_img.paste(pred_mesh_img, (gt_pc_img.width, gt_pc_img.height))
    
    combined_path = os.path.join(save_path, "comparison.png")
    combined_img.save(combined_path)
    return combined_img
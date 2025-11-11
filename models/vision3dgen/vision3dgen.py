import torch
import torch.nn as nn
import open3d as o3d
from .triplane_constructor import TriplaneConstructor
from .feature_alignment import FeatureAlignment
from .sdf_model import SDFModel
from .voxelizer import Voxelizer
from models.diff2dpoint.point_encoder import PointEncoder

class Vision3DGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.triplane_constructor = TriplaneConstructor(
            resolution=config["triplane"]["resolution"],
            num_channels=config["triplane"]["num_channels"],
            num_center_points=config["point_cloud_encoding"]["num_center_points"],
            k_neighbors=config["point_cloud_encoding"]["k_neighbors"],
            sigma=config["point_cloud_encoding"]["sigma"]
        )
        
        self.feature_alignment = FeatureAlignment(
            image_encoder=config["feature_alignment"]["image_encoder"],
            image_feat_dim=config["feature_alignment"]["image_feat_dim"],
            latent_dim=config["triplane"]["num_channels"]*8,
            sparse_resnet_blocks=config["feature_alignment"]["sparse_resnet_blocks"]
        )
        
        self.sdf_model = SDFModel(
            mlp_layers=config["sdf"]["mlp_layers"],
            mlp_hidden_dims=config["sdf"]["mlp_hidden_dims"],
            output_dim=config["sdf"]["output_dim"],
            activation=config["sdf"]["activation"]
        )
        
        self.voxelizer = Voxelizer(
            resolution=config["feature_alignment"]["mesh_voxel_resolution"],
            iso_surface_radius=config["feature_alignment"]["iso_surface_radius"]
        )
        
        self.point_encoder = PointEncoder(
            input_dim=3,
            hidden_dim=128,
            output_dim=256
        )

    def forward(self, img, point_cloud, mesh):
        batch_size = img.shape[0]
        device = img.device
        
        point_feat = self.point_encoder(point_cloud)
        latent_triplane = self.triplane_constructor(point_cloud)
        
        voxel_mesh = torch.cat([self.voxelizer(m) for m in mesh], dim=0)
        img_feat_aligned, mesh_feat_aligned, align_loss = self.feature_alignment(
            img, voxel_mesh, point_feat, latent_triplane
        )
        
        img_feat_proj = nn.Linear(img_feat_aligned.shape[-1], latent_triplane.shape[2])(img_feat_aligned).mean(dim=1).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        mesh_feat_proj = nn.Linear(mesh_feat_aligned.shape[-1], latent_triplane.shape[2])(mesh_feat_aligned).mean(dim=1).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        
        enhanced_triplane = latent_triplane + img_feat_proj + mesh_feat_proj
        
        coords = torch.randn(batch_size, 10000, 3, device=device).clamp(-1, 1)
        sdf = self.sdf_model(enhanced_triplane, coords)
        return sdf, align_loss

    def generate_3d_surface(self, img, point_cloud, mesh):
        self.eval()
        with torch.no_grad():
            point_feat = self.point_encoder(point_cloud)
            latent_triplane = self.triplane_constructor(point_cloud)
            
            voxel_mesh = torch.cat([self.voxelizer(m) for m in mesh], dim=0)
            img_feat_aligned, mesh_feat_aligned, _ = self.feature_alignment(
                img, voxel_mesh, point_feat, latent_triplane
            )
            
            img_feat_proj = nn.Linear(img_feat_aligned.shape[-1], latent_triplane.shape[2])(img_feat_aligned).mean(dim=1).unsqueeze(1).unsqueeze(3).unsqueeze(4)
            mesh_feat_proj = nn.Linear(mesh_feat_aligned.shape[-1], latent_triplane.shape[2])(mesh_feat_aligned).mean(dim=1).unsqueeze(1).unsqueeze(3).unsqueeze(4)
            
            enhanced_triplane = latent_triplane + img_feat_proj + mesh_feat_proj
            
            verts, faces, normals = self.sdf_model.extract_surface(
                enhanced_triplane,
                resolution=self.config["sdf"]["marching_cubes_resolution"],
                threshold=0.0
            )
            
            mesh_surface = o3d.geometry.TriangleMesh()
            mesh_surface.vertices = o3d.utility.Vector3dVector(verts)
            mesh_surface.triangles = o3d.utility.Vector3iVector(faces)
            mesh_surface.vertex_normals = o3d.utility.Vector3dVector(normals)
            return mesh_surface
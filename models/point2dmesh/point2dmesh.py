import torch
import torch.nn as nn
from .transformer_decoder import TransformerDecoder
from .dpo_trainer import DPOTrainer
from .mesh_generator import MeshGenerator
from models.diff2dpoint.point_encoder import PointEncoder

class Point2DMesh(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.point_encoder = PointEncoder(
            input_dim=3,
            hidden_dim=128,
            output_dim=config["transformer"]["point_feat_dim"]
        )
        
        self.transformer = TransformerDecoder(
            num_layers=config["transformer"]["num_layers"],
            hidden_dim=config["transformer"]["hidden_dim"],
            num_heads=config["transformer"]["num_heads"],
            dropout=config["transformer"]["dropout"],
            point_feat_dim=config["transformer"]["point_feat_dim"],
            vocab_size=config["transformer"]["mesh_token_vocab_size"]
        )
        
        self.dpo_trainer = DPOTrainer(
            model=self.transformer,
            temperature=config["dpo"]["temperature"]
        )
        
        self.mesh_generator = MeshGenerator(
            vertex_offset_range=config["mesh_generation"]["vertex_offset_range"],
            face_type=config["mesh_generation"]["face_connectivity_type"]
        )

    def forward(self, point_cloud, mesh_tokens=None):
        point_feat = self.point_encoder(point_cloud)
        if mesh_tokens is not None:
            logits = self.transformer(mesh_tokens[:, :-1], point_feat)
            return logits
        else:
            tokens = self.transformer.generate(
                point_feat,
                seq_len=self.config["transformer"]["seq_len"],
                sampling_strategy=self.config["mesh_generation"]["sampling_strategy"],
                nucleus_p=self.config["mesh_generation"]["nucleus_p"]
            )
            vertices, faces = self.mesh_generator.decode_tokens(tokens, point_feat)
            return vertices, faces

    def fine_tune_dpo(self, dataloader, optimizer):
        self.dpo_trainer.fine_tune(
            dataloader=dataloader,
            optimizer=optimizer,
            epochs=self.config["dpo"]["fine_tune_epochs"]
        )

    def generate_mesh(self, point_cloud, post_process=True):
        self.eval()
        with torch.no_grad():
            vertices, faces = self.forward(point_cloud)
            meshes = []
            for v, f in zip(vertices, faces):
                mesh = self.mesh_generator.generate_mesh(v, f)
                if post_process:
                    mesh = self.mesh_generator.post_process(mesh, smooth_iterations=self.config["inference"]["smooth_iterations"])
                meshes.append(mesh)
        return meshes
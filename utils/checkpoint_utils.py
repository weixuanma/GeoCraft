import os
import torch
import json

def save_checkpoint(model, optimizer, epoch, loss, save_dir, checkpoint_name="checkpoint"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{checkpoint_name}_epoch_{epoch:03d}.pth")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss

def resume_training(model, optimizer, save_dir, last_epoch=-1):
    if last_epoch == -1:
        checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
        if not checkpoint_files:
            raise ValueError("No checkpoint files found for resuming training")
        
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        last_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(save_dir, last_checkpoint)
    else:
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{last_epoch:03d}.pth")
    
    return load_checkpoint(model, optimizer, checkpoint_path)

def save_config(config, save_dir, config_name="config"):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, f"{config_name}.json")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    return config_path

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
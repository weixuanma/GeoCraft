import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from models.ccmim import CCMIM
from utils.data_utils import ConcreteDefectDataset
from utils.metric_utils import calculate_metrics
from utils.log_utils import Logger
from utils.vis_utils import save_detection_visualizations

def set_seed(seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_trained_model(config, device):
    model = CCMIM(
        mim_hidden_dim=config['model']['mim']['hidden_dim'],
        ddf_channel=config['model']['ddf']['channel'],
        spt_token_level=config['model']['spt']['token_level'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    checkpoint = torch.load(config['test']['ckpt_path'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def test_model(model, test_loader, metric_fn, device, config, logger):
    all_preds = []
    all_targets = []
    all_images = []
    all_image_paths = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Testing on {config['data']['dataset']}")
        for batch in pbar:
            images, targets, image_paths = batch
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            
            all_preds.extend(torch.softmax(outputs['detections'], dim=-1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            all_image_paths.extend(image_paths)
    
    metrics = metric_fn(
        all_preds,
        all_targets,
        iou_thres=config['test']['iou_thres'],
        conf_thres=config['test']['conf_thres']
    )
    
    log_msg = (f"Test Results on {config['data']['dataset']} | "
               f"Precision: {metrics['precision']:.4f} | "
               f"Recall: {metrics['recall']:.4f} | "
               f"F1-Score: {metrics['f1']:.4f} | "
               f"mAP50: {metrics['mAP50']:.4f} | "
               f"Miss Rate: {metrics['miss_rate']:.4f} | "
               f"False Positive Rate: {metrics['fp_rate']:.4f}")
    logger.write(log_msg)
    print(log_msg)
    
    return metrics, all_images, all_preds, all_image_paths

def save_test_results(metrics, config, dataset_name):
    result_dir = config['test']['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    result_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Precision': [metrics['precision']],
        'Recall': [metrics['recall']],
        'F1-Score': [metrics['f1']],
        'mAP50': [metrics['mAP50']],
        'Miss Rate': [metrics['miss_rate']],
        'False Positive Rate': [metrics['fp_rate']],
        'Image Size': [config['data']['image_size']],
        'Confidence Threshold': [config['test']['conf_thres']],
        'IoU Threshold': [config['test']['iou_thres']]
    })
    
    result_path = os.path.join(result_dir, f"{dataset_name}_test_results.csv")
    if os.path.exists(result_path):
        existing_df = pd.read_csv(result_path)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    result_df.to_csv(result_path, index=False)
    
    with open(os.path.join(result_dir, f"{dataset_name}_test_metrics.txt"), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

def main(config_path):
    config = load_config(config_path)
    set_seed(config['test']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(config['test']['log_dir'], exist_ok=True)
    os.makedirs(config['test']['result_dir'], exist_ok=True)
    os.makedirs(config['test']['vis_dir'], exist_ok=True)
    
    logger = Logger(os.path.join(config['test']['log_dir'], "test_log.txt"))
    logger.write(f"Test Configuration: {config}")
    logger.write(f"Using Device: {device}")
    
    model = load_trained_model(config, device)
    logger.write("Trained model loaded successfully")
    
    test_dataset = ConcreteDefectDataset(
        data_dir=os.path.join(config['data']['root_dir'], config['data']['dataset'], "test"),
        image_size=config['data']['image_size'],
        augment=False,
        return_path=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test']['batch_size'],
        shuffle=False,
        num_workers=config['test']['num_workers'],
        pin_memory=True
    )
    
    logger.write(f"Test Dataset Loaded: {config['data']['dataset']} (Total Samples: {len(test_dataset)})")
    
    metrics, all_images, all_preds, all_image_paths = test_model(
        model, test_loader, calculate_metrics, device, config, logger
    )
    
    save_test_results(metrics, config, config['data']['dataset'])
    logger.write(f"Test results saved to {config['test']['result_dir']}")
    
    save_detection_visualizations(
        images=all_images,
        preds=all_preds,
        image_paths=all_image_paths,
        save_dir=config['test']['vis_dir'],
        conf_thres=config['test']['conf_thres'],
        class_names=config['model']['class_names']
    )
    logger.write(f"Detection visualizations saved to {config['test']['vis_dir']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test CCMIM Model for Concrete Defect Detection")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml", help="Path to test config file")
    args = parser.parse_args()
    main(args.config)
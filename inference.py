import os
import yaml
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

from models.ccmim import CCMIM
from utils.vis_utils import draw_detection_boxes
from utils.log_utils import Logger

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

def load_inference_model(config, device):
    model = CCMIM(
        mim_hidden_dim=config['model']['mim']['hidden_dim'],
        ddf_channel=config['model']['ddf']['channel'],
        spt_token_level=config['model']['spt']['token_level'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    checkpoint = torch.load(config['inference']['ckpt_path'], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def prepare_image(image_path, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, image, original_size

def infer_single_image(model, image_path, config, device, logger):
    image_tensor, original_image, original_size = prepare_image(
        image_path, config['inference']['image_size'], device
    )
    
    with torch.no_grad():
        outputs = model(image_tensor)
        preds = torch.softmax(outputs['detections'], dim=-1).cpu().numpy()[0]
    
    detected_boxes = []
    detected_scores = []
    detected_classes = []
    
    for idx, pred in enumerate(preds):
        conf_score = pred[1]
        if conf_score >= config['inference']['conf_thres']:
            x1, y1, x2, y2 = pred[2:6]
            x1 = int(x1 * original_size[0])
            y1 = int(y1 * original_size[1])
            x2 = int(x2 * original_size[0])
            y2 = int(y2 * original_size[1])
            
            detected_boxes.append([x1, y1, x2, y2])
            detected_scores.append(conf_score)
            detected_classes.append(0)
    
    vis_image = draw_detection_boxes(
        original_image,
        detected_boxes,
        detected_scores,
        detected_classes,
        class_names=config['model']['class_names'],
        box_color=(0, 255, 0),
        text_color=(255, 0, 0)
    )
    
    save_path = os.path.join(
        config['inference']['output_dir'],
        f"infer_{os.path.basename(image_path)}"
    )
    cv2.imwrite(save_path, cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR))
    
    log_msg = f"Inference completed for {image_path} | Detected cracks: {len(detected_boxes)} | Saved to {save_path}"
    logger.write(log_msg)
    print(log_msg)
    
    return {
        'image_path': image_path,
        'detected_cracks': len(detected_boxes),
        'confidence_scores': detected_scores,
        'bounding_boxes': detected_boxes,
        'output_path': save_path
    }

def infer_batch_images(model, image_dir, config, device, logger):
    image_paths = [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    results = []
    pbar = tqdm(image_paths, desc=f"Batch Inference on {image_dir}")
    for img_path in pbar:
        result = infer_single_image(model, img_path, config, device, logger)
        results.append(result)
        pbar.set_postfix({"Processed": len(results), "Total": len(image_paths)})
    
    return results

def main(config_path):
    config = load_config(config_path)
    set_seed(config['inference']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(config['inference']['output_dir'], exist_ok=True)
    os.makedirs(config['inference']['log_dir'], exist_ok=True)
    
    logger = Logger(os.path.join(config['inference']['log_dir'], "inference_log.txt"))
    logger.write(f"Inference Configuration: {config}")
    logger.write(f"Using Device: {device}")
    
    model = load_inference_model(config, device)
    logger.write("Inference model loaded successfully")
    
    if config['inference']['input_type'] == 'single':
        infer_single_image(model, config['inference']['input_path'], config, device, logger)
    elif config['inference']['input_type'] == 'batch':
        infer_batch_images(model, config['inference']['input_dir'], config, device, logger)
    else:
        raise ValueError("Invalid input_type: choose 'single' for one image or 'batch' for multiple images")
    
    logger.write("Inference process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for CCMIM Concrete Defect Detection Model")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml", help="Path to inference config file")
    args = parser.parse_args()
    main(args.config)
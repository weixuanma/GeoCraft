import cv2
import numpy as np
from PIL import Image

def draw_detection_boxes(image, bboxes, scores, cls_ids, class_names, box_color=(0, 255, 0), text_color=(255, 0, 0)):
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        
        cls_name = class_names[cls_id] if cls_id < len(class_names) else "Unknown"
        text = f"{cls_name}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1
        text_y = y1 - 10 if y1 > 10 else y1 + text_size[1] + 5
        
        cv2.rectangle(
            image,
            (text_x, text_y - text_size[1]),
            (text_x + text_size[0], text_y),
            box_color,
            -1
        )
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1
        )
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

def plot_loss_curve(train_losses, val_losses, save_path, title="Loss Curve"):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", color="red", linewidth=2)
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_detection_visualizations(images, preds, image_paths, save_dir, conf_thres, class_names):
    os.makedirs(save_dir, exist_ok=True)
    for img, pred, img_path in zip(images, preds, image_paths):
        img = np.transpose(img, (1, 2, 0))
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        
        bboxes = []
        scores = []
        cls_ids = []
        for p in pred:
            if p[1] >= conf_thres:
                bboxes.append(p[2:6])
                scores.append(p[1])
                cls_ids.append(0)
        
        vis_img = draw_detection_boxes(img, bboxes, scores, cls_ids, class_names)
        save_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, save_name)
        vis_img.save(save_path)
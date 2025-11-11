import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_metrics(all_preds, all_targets, iou_thres=0.5, conf_thres=0.5):
    filtered_preds = [pred for pred in all_preds if pred[1] >= conf_thres]

    y_true = []
    y_pred = []
    for target, pred in zip(all_targets, filtered_preds):
        y_true.append(1 if len(target["cls_ids"]) > 0 else 0)
        y_pred.append(1 if len(pred) > 0 else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    mAP50 = _calculate_map50(filtered_preds, all_targets, iou_thres)

    miss_rate = 1 - recall
    fp_rate = sum([1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0]) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": mAP50,
        "miss_rate": miss_rate,
        "fp_rate": fp_rate
    }

def _calculate_map50(preds, targets, iou_thres):
    if len(preds) == 0 or len(targets) == 0:
        return 0.0

    coco_gt = COCO()
    coco_dt = COCO()

    images = []
    annotations_gt = []
    annotations_dt = []
    img_id = 0
    anno_id = 0

    for target in targets:
        images.append({"id": img_id})
        for bbox, cls_id in zip(target["bboxes"], target["cls_ids"]):
            annotations_gt.append({
                "id": anno_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": bbox,
                "area": (bbox[2]-bbox[0])*(bbox[3]-bbox[1]),
                "iscrowd": 0
            })
            anno_id += 1
        img_id += 1

    coco_gt.dataset = {"images": images, "annotations": annotations_gt, "categories": [{"id": 0, "name": "crack"}]}
    coco_gt.createIndex()

    img_id = 0
    for pred in preds:
        if len(pred) < 6:
            img_id += 1
            continue
        conf = pred[1]
        bbox = pred[2:6]
        annotations_dt.append({
            "id": anno_id,
            "image_id": img_id,
            "category_id": 0,
            "bbox": bbox,
            "score": conf,
            "area": (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        })
        anno_id += 1
        img_id += 1

    coco_dt.dataset = {"images": images, "annotations": annotations_dt, "categories": [{"id": 0, "name": "crack"}]}
    coco_dt.createIndex()

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [iou_thres]
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[1]
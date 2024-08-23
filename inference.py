from mmdet.apis import inference_detector, init_detector
from mmdet.core.evaluation.mean_ap import *
import cv2
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, help='Example: --device cuda:0', required=False, default='cpu')
parser.add_argument('--data_path', type=str, help='Example: data/', required=True, default='data/')
parser.add_argument('--model_path', type=str, help='Model path', required=False, default='models/detr.pth')
parser.add_argument('--config_path', type=str, help='Config path', required=False, default='configs/detr_config.py')
parser.add_argument('--result_path', type=str, help='Result path', required=False, default='results/')

args = parser.parse_args()

config_path = args.config_path
model_path = args.model_path
device = args.device
data_path = args.data_path
result_path = args.result_path

if not os.path.exists(result_path):
    os.mkdir(result_path)

def apply_nms(result, image_path, class_size, iou_thr=0.1, scr_thr=0.2):
    """
    Apply NMS for detection results.

    Parameters
    ----------
    result: list
        Model detection result.
    image_path: str
        Image path.
    class_size: int
        Number of classes.
    iou_thr: float
        IoU threshold.
    scr_thr: float
        Confidence threshold.

    Returns
    -------
    nms_results: list
        NMS-applied results.
    """
    
    img = Image.open(image_path)
    img = np.array(img)
    
    labels_list, scores_list, boxes_list = get_labels_scores_boxes_list([result], img.shape[1], img.shape[0], class_size)
    
    # Ensure non-empty results before applying NMS
    if len(boxes_list) > 0 and len(scores_list) > 0 and len(labels_list) > 0:
        try:
            b, s, l = nms(boxes_list[0], scores_list[0], labels_list[0], iou_thr)
        except ValueError:
            b, s, l = [], [], []
    else:
        b, s, l = [], [], []
    
    n_boxes = [de_normalize_bbox(b[i], img.shape[1], img.shape[0]) for i in range(len(b))]
    
    label_dict = {str(i): [] for i in range(class_size)}
    
    for i in range(len(n_boxes)):
        if s[i] > scr_thr:
            n_boxes[i].append(s[i])
            label_dict[str(l[i])].append(np.array(n_boxes[i]))
    
    nms_results = []
    for key, value in label_dict.items():           
        r = np.array(value)
        if len(r) == 0:
            r = np.zeros((0, 5)).astype('float32')
        nms_results.append(r)
    
    return nms_results

def get_labels_scores_boxes_list(results, shapex, shapey, class_size):
    """
    Extraction of label, score, box lists in accordance with 
    each image of the result list obtained in the model output.


    Parametreler
    ----------    
    results: list
        List of model results
    shapex: int
        image weight
    shapey: int
        image height
    class_size : int
        length of label names

    Retget_gtbboxurns
    -------
    all_labels_list: list
        list of labels
    all_scores_list: list
        list of scores
    all_boxes_list: list
        list of boxes

    """
    
    all_labels_list = []
    all_scores_list = []
    all_boxes_list = []
    for j in range(len(results)):
        labels_list = []
        scores_list = []
        boxes_list = []
        for i in range(class_size):
            if(results[j][i].shape[0]>0):
                [labels_list.append(i) for k in range(results[j][i].shape[0])]
                scores_list.append(results[j][i][:,-1])
                for k in results[j][i][:,:-1]:
                    boxes_list.append(normalize_bbox(k, shapex, shapey))
        if len(scores_list) > 0: 
            all_labels_list.append(np.array(labels_list))
            all_scores_list.append(np.concatenate(scores_list))
            all_boxes_list.append(boxes_list)
    return all_labels_list,all_scores_list,all_boxes_list

def normalize_bbox(bbox, shapex, shapey):
    """
    Process of normalizing the bbox points of the image for NMS.

    Parametreler
    ----------    
    bbox: list
        images' bounding box

    Returns
    -------
    bbox: list
        normalized bounding box

    """
    bbox = bbox.copy()
    bbox[0] /= shapex
    bbox[1] /= shapey
    bbox[2] /= shapex
    bbox[3] /= shapey
    return bbox

def nms(boxes, scores, labels, iou_thr=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on the bounding boxes.
    
    Parameters
    ----------
    boxes : list of lists
        Bounding boxes [x1, y1, x2, y2] in normalized coordinates.
    scores : list
        Confidence scores for each bounding box.
    labels : list
        Class labels for each bounding box.
    iou_thr : float, optional
        Intersection over Union (IoU) threshold for suppression, by default 0.5.
    
    Returns
    -------
    list of boxes, list of scores, list of labels
        Bounding boxes, scores, and labels after NMS.
    """
    
    # Convert boxes to a numpy array
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Initialize a list to hold the indices of the boxes to keep
    keep = []
    
    # Sort the boxes by score in descending order
    indices = np.argsort(scores)[::-1]
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        # Calculate IoU of the current box with the rest
        iou = compute_iou(boxes[current], boxes[indices[1:]])
        
        # Remove boxes with IoU higher than the threshold
        indices = indices[1:][iou <= iou_thr]
    
    return boxes[keep].tolist(), scores[keep].tolist(), labels[keep].tolist()

def compute_iou(box, boxes):
    """
    Compute IoU between a box and a list of boxes.
    
    Parameters
    ----------
    box : list
        A single bounding box [x1, y1, x2, y2].
    boxes : list of lists
        List of bounding boxes [x1, y1, x2, y2].
    
    Returns
    -------
    iou : numpy array
        Array of IoU values.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    iou = intersection / (box_area + boxes_area - intersection)
    
    return iou

def de_normalize_bbox(bbox, shapex, shapey):
    """
    Process of denormalizing the normalized bbox points of the image for NMS.
    

    Parametreler
    ---------- 
    bbox: list
        normalized bbox

    Returns
    -------
    bbox: list
        bbox

    """
    bbox = bbox.copy()
    bbox[0] *= shapex
    bbox[1] *= shapey
    bbox[2] *= shapex
    bbox[3] *= shapey
    return bbox

model = init_detector(config_path, model_path, device)

for img_name in os.listdir(data_path):
    if img_name.split('.')[1] != 'png':
        continue
    img_path = os.path.join(data_path, img_name)
    result = inference_detector(model, img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    result = apply_nms(result, img_path, 1)

    for coordinates in result[0]:
        xmin,ymin,xmax,ymax = int(coordinates[0]),int(coordinates[1]),int(coordinates[2]), int(coordinates[3])
        img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255, 0, 0 ), thickness=8)
        img = cv2.putText(img, "{:.2f}".format(coordinates[4]), (xmin,ymin-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.5, color=(255, 0, 0), thickness = 3)

    cv2.imwrite(os.path.join(result_path, img_name), img)
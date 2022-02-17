import yolov5
import numpy as np

id_categories = {1: 'Insulating material',
 4: 'Drum',
 2: 'Bottle-shaped',
 3: 'Can-shaped',
 5: 'Other packaging',
 6: 'Tire',
 7: 'Fishing net / cord',
 8: 'Easily namable',
 9: 'Unclear',
 0: 'Sheet / tarp / plastic bag / fragment'}

categories_id = {v: k for k, v in id_categories.items()}
get_id = lambda cat: categories_id[cat]

def load_model(model_path, device, conf=0.35, iou=0.50):
    model = yolov5.load(model_path, device=device)
    model.conf = conf
    model.iou  = iou
    model.classes = None
    model.multi_label = False
    model.max_det = 1000
    return model

def voc2centerdims(bboxes):
    """
    voc  => [x1, y1, x2, y2]
    output => [xcenter, ycenter, w, h]
    """
    bboxes[..., 2:4] -= bboxes[..., 0:2] # find w,h
    bboxes[..., 0:2] += bboxes[...,2:4]/2 # find center
    return bboxes

def predict_yolo(model, img, size=768, augment=False):
    """
    interpret yolo prediction object
    """
    results = model(img, size=size, augment=augment)
    preds   = results.pandas().xyxy[0]
    bboxes  = preds[['xmin','ymin','xmax','ymax']].values # voc format
    if len(bboxes):
        bboxes = voc2centerdims(bboxes)
        bboxes  = bboxes.astype(int)
        confs   = preds.confidence.values
        labels  = np.array(list(map(get_id, preds.name.values)))
        return bboxes, confs, labels
    else:
        return np.array([]), np.array([]), np.array([])

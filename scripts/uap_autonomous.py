import torch, torchvision
from torchvision import transforms
print(torch.__version__, torch.cuda.is_available())

# import some common libraries
import numpy as np
from PIL import Image
import os, glob, cv2, pickle
from pycocotools.coco import COCO

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

def get_detector(config_file, nms_thresh, data_type, weight_dir=None, dataset_name=None):
  """ for each image you make an prediction and find the indexes of correctly classified rois
  
  returns: 
      - model: object detector(nn.module)
      - cfg: config file for the module

  params:
      - config_file:
      - nms_thresh: it determins the minimum confidence threshold for rois to survive for NMS
      - data_type: 
      - weight_file: 
  """
  cfg = get_cfg()

  # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
  cfg.merge_from_file(model_zoo.get_config_file(config_file))

  # set threshold for this model
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = nms_thresh  

  # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
  if weight_dir is None:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file) # use the pretrained weights

  else:
    cfg.MODEL.WEIGHTS = os.path.join(weight_dir, 'model_final.pth') # use the custom trained weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(dataset_name).thing_classes)  # number of classes 

  if data_type == 'model':
    model = build_model(cfg)  # returns a torch.nn.Module
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

  if data_type == 'predictor':
    model = DefaultPredictor(cfg)

  return model, cfg


class SubCocoDetection(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
    root (string): Root directory where images are downloaded to.
    annFile (string): Path to json annotation file.
    transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.ToTensor``
    target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """

    def __init__(self, root, annFile, classes, transform=None, target_transform=None, transforms=None):
        # super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        self.coco = COCO(annFile) # load entire coco annotation file
        self.transforms = transforms
        self.ids = sorted(self.coco.getImgIds(catIds=self.coco.getCatIds(catNms=classes))) # get list of image ids
        print("number of images in {} = {}".format(classes, len(self.ids)))

    def __getitem__(self, index):
        """
        Args:
            index (int): index

        Returns:
            img, target (tuple): target is the object returned by ``coco.loadAnns``.
        """
        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
        coco = self.coco 
        img_id = self.ids[index] # get one image id according to the index
        ann_ids = coco.getAnnIds(imgIds=img_id) # load annotation ids related to that image id
        coco_annotation = coco.loadAnns(ann_ids) # load annotation labels in that image id from annotation ids
      
        path = coco.loadImgs(img_id)[0]['file_name'] # load image annotation related to that image id
        img = Image.open(os.path.join(self.root, path)).convert('RGB') # open the PIL image
        
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []

        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(thing_dataset_id_to_contiguous_id[coco_annotation[i]['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])    
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["iscrowd"] = iscrowd

        # preprocess image if wanted
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.Resize((640, 480)))
    return transforms.Compose(custom_transforms)


def filter_pos_pred(dataset, model, cls, cfg, max_img):
    """ to completely isolate the effect of the perturbation, we have to only use the images that the detector results in correct prediction
    
    returns: 
        - train_data: list[tuple], images that have at least one correct prediction of our target class

    params:
        - dataset: the dataset that we want to filter out
        - model: the object detector we want to test on
        - cls: the target class that we are interested in
        - cfg: config file that contains class category strings for COCO2017 train
        - max_img: max number of images that we want to keep as training data
    """

    # split into train and val data
    train_data = []
    gt_total_count = 0
    pred_total_count = 0
    img_num = 0

    for i in range(len(dataset)):
        is_correct, gt_count, pred_count = is_pos_pred(dataset[i], model, cls, cfg)
        
        if is_correct is True:
            train_data.append(dataset[i])
            img_num += 1
            gt_total_count += gt_count
            pred_total_count += pred_count

            # finish if collected images exceed the desired number of training examples
            if img_num >= max_img:
                break
            
    print('ground truth # of classes for filtered images: {}, predicted # of classes for filtered images: {}'.format(gt_total_count, pred_total_count))
    print('number of filtered images: {}'.format(len(train_data)))
    return train_data


def is_pos_pred(data, model, cls, cfg):
    """ for each image, you check the number of ground truth and predicted objects from our model, then determine if the prediction finds at least one object of interest
    
    returns: 
        - is_pos_pred: boolean, the prediction finds at least one correct object detection of interest
        - gt_count: int, the number of ground truth objects of our interest in the image
        - pred_count: int, the number of predicted objects of our interest in the image

    params:
        - data:
        - model: the object detector we want to test on
        - cls: the target class that we are interested in
        - cfg: config file that contains class category strings for COCO2017 train
    """
       
    # convert image PIL to torch tensor 
    img, target = data
    
    pred_count = 0
    gt_count = np.sum((target["labels"].numpy() == cls))
    is_pos_pred = False
    
    img = np.array(img)
    img_t = torch.from_numpy(img).float()
    img_t = img_t.permute(2, 0, 1)

    # add universal adversarial perturbation to clean image
    img_dict = [dict(image=img_t)]

    # run detection on the model and get the confidence scores of the detections
    outputs = model(img_dict)[0]
    outputs = outputs['instances']
    outputs.pred_boxes.tensor = torch.tensor(outputs.pred_boxes.tensor, requires_grad=False)

    # show ground truth annotations of the image
    # check_annotations(data_dict, cfg)

    # show predictions of the image
    # v = Visualizer(cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2RGB), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs.to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])

    # for each instance prediction, 
    for idx, pred_cls in enumerate(outputs.pred_classes):

        # check that predicted class is the target class 
        if pred_cls.item() == cls:

            # compare it with all ground truth boxes of the target class
            for i, gt_labels in enumerate(target["labels"]):

                # if the gt box is the target class, compute their iou to see if it's true positive
                if gt_labels.item() == pred_cls:
                    gt_box = target['boxes'][i]

                    if bb_intersection_over_union(outputs.pred_boxes.tensor[idx].cpu().numpy(), gt_box.numpy()) > 0.5:
                        is_pos_pred = True
                        pred_count += 1
                        break
                        
    return is_pos_pred, gt_count, pred_count


def bb_intersection_over_union(boxA, boxB):
    """ for each image you make an prediction and find the indexes of correctly classified rois
    
    returns: 
        - iou: float, intersection over union between two bounding boxes

    params:
        - boxA: 
        - boxB:
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def loss_fn(outputs, uap, cls, p=np.inf):
    """ the objective function aims to lower the confidence scores of the objects of target class in the detection results
    while keep the perturbatio norm as small as possible
    
    returns: 
        - loss: loss value for the prediction of the image

    params:
        - outputs:
        - uap:
        - cls:
    """
    
    scores = outputs['instances'].scores
   
    # when the target is the entire set of classes
    if cls == -1:
        loss = scores.sum()
        if p == np.inf:
            loss += torch.max(torch.abs(uap))
        elif p == 1:
            loss += torch.mean(torch.abs(uap))

        return loss
        
    # when the target is a specific class
    else:
        pred_classes = outputs['instances'].pred_classes
        loss = scores[pred_classes == cls].sum()
        if p == np.inf:
            loss += torch.max(torch.abs(uap))
        elif p == 1:
            loss += torch.mean(torch.abs(uap))

        return loss


def udos(cfg, model, dataset, n_epochs, Xi, epsilon, cls=-1):
    """create a universal perturbation that attacks a set of images at once

    returns: 
        - uap: loss value for the prediction of the image
        - metric_ls: 

    params:
        - cfg:
        - model:
        - datsets:
        - n_epochs:
        - Xi:
        - epsilon:
        - cls:
    """

    # initialize variables
    metric_ls = [] # list of lists storing metrics for each epoch
    n = len(dataset) # of images in the attack set I
    uap = torch.zeros(1) # universal perturbation that we want to compute from the attack set I
    v = torch.zeros(len(dataset), 3, 640, 480, dtype=torch.float) # gradient computed from each image to update universal perturbation
    
    # for each epoch
    for epoch in range(n_epochs):
        print("epoch {}".format(epoch+1))

        # (re)initialize metrics: loss, instance-level blind degree, image-level blind degree
        inst_blind_deg = 0
        img_blind_deg = 0
        total_loss = 0
        
        # for each image
        for i, data in enumerate(dataset):

            # convert image to the format list[dict] that model accepts as input
            img, _ = data
            img = np.array(img) # convert it to numpy
            img_t = torch.from_numpy(img).float() # convert numpy to torch tensor
            img_t = img_t.permute(2, 0, 1) # (H, W, C) to (C, H, W)
            img_t.requires_grad = True # set True to calculate gradient for this tensor

            img_dict = [dict(image=img_t+uap)] # add perturbation to clean image

            outputs = model(img_dict)[0] # run detection on the model

            # get confidence scores and class predictions on all the detection instances
            scores = outputs['instances'].scores
            pred_classes = outputs['instances'].pred_classes
            
            # given that current perturbation haven't failed the detector to detect no objects in this image, we keep optimizing the perturbation for this image
            if len(scores[pred_classes == cls]) != 0:
                train_loss = loss_fn(outputs, uap, cls, 1)
                
                model.zero_grad() # zero all existing gradients
                img_dict[0]['image'].retain_grad() # this is needed as the image is not in the leaf-node

                # Calculate gradients of model through backward pass
                train_loss.backward()

                # update v_i by step size multipled by the gradient of loss w.r.t to perturbed_image
                v[i] -= epsilon * img_dict[0]['image'].grad.data
                            
                # project the updated universal perturbation to the nearest point that satisfies the constraint Xi
                uap = torch.clamp(uap + v[i], min=-Xi, max=Xi)

                # calculate metrics
                inst_blind_deg = inst_blind_deg + (float(train_loss) - torch.mean(torch.abs(uap)))
                img_blind_deg = img_blind_deg + 1 # add image-level if the detector finds at least one object beyond score threhold
                total_loss = total_loss + float(train_loss)
                
        # calculate average metrics for this epoch
        inst_blind_deg = inst_blind_deg / n
        img_blind_deg = img_blind_deg / n
        total_loss = total_loss / n
        print("loss: {}, perturbation l1-norm: {}, instance_level_blind_deg: {}, img_level_blind_deg: {}".format(total_loss, torch.mean(torch.abs(uap)), inst_blind_deg, img_blind_deg))
        metric_ls.append((total_loss, torch.mean(torch.abs(uap)), inst_blind_deg, img_blind_deg)) # append metric for this epoch to this list for later visualization


    return uap, metric_ls


if __name__ == "__main__":
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    root = '/fs/vulcan-datasets/coco/images/train2017'
    annFile = '/fs/vulcan-datasets/coco/annotations/instances_train2017.json'
    n_epochs = 50
    Xi = 10 # [5, 10, 15, 20, 25]
    eps = 1 # [1, 5, 10, 20, 30] 
    n_imgs = 500
    score_thresh = 0.7
    iou_thresh = 0.5

    # load object detector you want to test
    model, cfg = get_detector(config_file, score_thresh, data_type='model')

    # set names and indexs of the target classes that you want to experiment
    cls_strs = [['stop sign'], ['traffic light'], ['truck'], ['car'], ['person']]
    cls_idxs = [11, 9, 7, 2, 0] # person=0, car=2, truck=7, traffic light=9, stop sign=11

    # for each class of interest
    for cls_str, cls_idx in zip(cls_strs, cls_idxs):

        # load images that at least one instance of that class exist
        coco = SubCocoDetection(root=root, annFile=annFile, classes=cls_str, transforms=get_transform())
        
        # filter images that at least one prediction of the instance was correct
        train_data = filter_pos_pred(coco, model, cls_idx, cfg, n_imgs)

        # run U-DOS algorithm to calculate universal perturbation and the evaluation result
        print("target class: {}, number of epochs: {}, Xi: {}, epsilon: {}, number of training imgs: {}".format(cls_str, n_epochs, Xi, eps, len(train_data)))
        uap, metric_ls = udos(cfg=cfg, model=model, dataset=train_data, n_epochs=n_epochs, Xi=Xi, epsilon=eps, cls=cls_idx)
        
        # save metric results to a pickle file
        metrics = np.array(metric_ls) # convert list to numpy array
        file_name = 'cls{}_epochs{}_xi{}_eps{}_imgs{}.pkl'.format(cls_str, n_epochs, Xi, eps, len(train_data))
        open_file = open(file_name, "wb")
        pickle.dump(metrics, open_file)
        open_file.close()


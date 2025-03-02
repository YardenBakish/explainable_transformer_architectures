import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import os
from misc.helper_functions import *


from tqdm import tqdm
from utils.metrices import *
from models.model_handler import model_env 
#from utils.saver import Saver
from utils.iou import IoU
import config
from data.imagenet_new import Imagenet_Segmentation

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from ViT_explanation_generator import Baselines, LRP
from old.model import deit_tiny_patch16_224 as vit_LRP
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F

plt.switch_backend('agg')


#changehere
#NUM_STEPS = 5
NUM_STEPS = 8
# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')

parser.add_argument('--custom-trained-models1', nargs='+', required=True, help='List of model paths for variant 1')
parser.add_argument('--custom-trained-models2', nargs='+', required=True, help='List of model paths for variant 2')

parser.add_argument('--otsu-thr', action='store_true')


parser.add_argument('--variant1', default = 'basic', help="")
parser.add_argument('--variant2', default = 'basic', help="")
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)
parser.add_argument('--output-dir', required = True)


parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
                    choices=[ 'rollout', 'lrp','transformer_attribution', 'full_lrp', 'lrp_last_layer',
                              'attn_last_layer', 'attn_gradcam', 'custom_lrp'],
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-ia', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fgx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-m', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-reg', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--is-ablation', type=bool,
                    default=False,
                    help='')

parser.add_argument('--data-path', type=str,
                     
                        help='')
parser.add_argument('--imagenet-seg-path', type=str, required=True)



args1 = parser.parse_args()
args2 = parser.parse_args()


args1.variant = args1.variant1
args2.variant = args2.variant2


args1.custom_trained_models = args1.custom_trained_models1
args2.custom_trained_models = args2.custom_trained_models2



config.get_config(args1, skip_further_testing = True)
config.set_components_custom_lrp(args1)

config.get_config(args2, skip_further_testing = True)
config.set_components_custom_lrp(args2)


os.makedirs(args1.output_dir, exist_ok=True)


args1.checkname = args1.method + '_' + args1.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")



imagenet_trans = transforms.Compose([
        #transforms.Resize(sizeX, interpolation=3),
        #transforms.CenterCrop(args.input_size), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])



def otsu_threshold(img):
    """
    Compute Otsu's threshold for a 2D array.
    """
    # Flatten the image into 1D array
    flat = img.flatten()
    
    # Get histogram
    hist, bins = np.histogram(flat, bins=256, range=(0,1))
    hist = hist.astype(float)
    
    # Get bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Get total number of pixels
    total = hist.sum()
    
    best_thresh = 0
    best_variance = 0
    
    # Calculate cumulative sums
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Calculate cumulative means
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    
    # Calculate between class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Get threshold with maximum variance
    idx = np.argmax(variance)
    best_thresh = bin_centers[idx]
    
    return best_thresh



test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation(args1.imagenet_seg_path,
                           transform=imagenet_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

# Model

#FIXME: currently only attribution method is tested. Add support for other methods using other variants 
model = vit_LRP(pretrained=True).cuda()
baselines = Baselines(model)


variant1_lrp_lst = []
variant2_lrp_lst = []

for elem in args1.custom_trained_models:
    

    if elem != None:
        if args1.data_set == 'IMNET100':
            args1.nb_classes = 100
        else:
            args1.nb_classes = 1000

        model_LRP = model_env(pretrained=False, 
                        args = args1,
                        hooks = True,
                    )

        checkpoint = torch.load(elem, map_location='cpu')

        model_LRP.load_state_dict(checkpoint['model'], strict=False)
        model_LRP.to(device)


    model_LRP.eval()
    lrp = LRP(model_LRP)

    variant1_lrp_lst.append(lrp)


for elem in args2.custom_trained_models:
    

    if elem != None:
        if args2.data_set == 'IMNET100':
            args2.nb_classes = 100
        else:
            args2.nb_classes = 1000

        model_LRP = model_env(pretrained=False, 
                        args = args2,
                        hooks = True,
                    )

        checkpoint = torch.load(elem, map_location='cpu')

        model_LRP.load_state_dict(checkpoint['model'], strict=False)
        model_LRP.to(device)


    model_LRP.eval()
    lrp = LRP(model_LRP)

    variant2_lrp_lst.append(lrp)



# orig LRP
model_orig_LRP = vit_LRP(pretrained=True).cuda()
model_orig_LRP.eval()
orig_lrp = LRP(model_orig_LRP)

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

model.eval()


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # pred[0, 0] = 282
    # print('Pred cls : ' + str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.cuda()

    return Tt


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()
    # Save input image


    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)
    
    # segmentation test for the rollout baseline
    if current_args.method    == 'custom_lrp':
        Res = current_lrp.generate_LRP(image.cuda(), method="custom_lrp", cp_rule = current_args.cp_rule).reshape(14, 14).unsqueeze(0).unsqueeze(0) 
    
    elif current_args.method == 'rollout':
        Res = baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the LRP baseline (this is full LRP, not partial)
    elif current_args.method == 'full_lrp':
        Res = orig_lrp.generate_LRP(image.cuda(), method="full", cp_rule = current_args.cp_rule).reshape(batch_size, 1, 224, 224)
    
    # segmentation test for our method
    elif current_args.method == 'transformer_attribution':
        Res = current_lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution", cp_rule = current_args.cp_rule).reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the partial LRP baseline (last attn layer)
    elif current_args.method == 'lrp_last_layer':
        Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer", is_ablation=current_args.is_ablation, cp_rule = current_args.cp_rule)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the raw attention baseline (last attn layer)
    elif current_args.method == 'attn_last_layer':
        Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer_attn", is_ablation=current_args.is_ablation, cp_rule = current_args.cp_rule)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the GradCam baseline (last attn layer)
    elif current_args.method == 'attn_gradcam':
        Res = baselines.generate_cam_attn(image.cuda()).reshape(batch_size, 1, 14, 14)

    if current_args.method != 'full_lrp':
        # interpolate to full image size (224,224)
        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
    
    # threshold between FG and BG is the mean    
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    
    batch_inter_lst, batch_union_lst, batch_correct_lst, batch_label_lst = [0 for i in range(NUM_STEPS)], [0 for i in range(NUM_STEPS)], [0 for i in range(NUM_STEPS)], [0 for i in range(NUM_STEPS)]
    batch_ap_lst, batch_f1_lst = [0 for i in range(NUM_STEPS)], [0 for i in range(NUM_STEPS)]

    bg_intersection_lst = [0 for i in range(NUM_STEPS)] #batch_inter[0]
    fg_intersection_lst = [0 for i in range(NUM_STEPS)] #batch_inter[1]
    tot_labeled_foreground_lst = [0 for i in range(NUM_STEPS)] #np.sum(labels[0].cpu().numpy() > 0)

    for i in range(NUM_STEPS):


        #changehere
        #ret =  Res.mean() + (i* 0.2)*(Res.max() - Res.mean())
      
        ret =  Res.mean() / ((i+1) * Res.std() + 1e-10 )


        Res_1 = Res.gt(ret).type(Res.type())
        Res_0 = Res.le(ret).type(Res.type())

        Res_1_AP = Res
        Res_0_AP = 1-Res

        Res_1[Res_1 != Res_1] = 0
        Res_0[Res_0 != Res_0] = 0
        Res_1_AP[Res_1_AP != Res_1_AP] = 0
        Res_0_AP[Res_0_AP != Res_0_AP] = 0



        pred = Res.clamp(min=current_args.thr) / Res.max()
        pred = pred.view(-1).data.cpu().numpy()
        target = labels.view(-1).data.cpu().numpy()

        output = torch.cat((Res_0, Res_1), 1)
        output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)


        correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
        inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
        batch_correct_lst[i] += correct
        batch_label_lst[i] += labeled
        batch_inter_lst[i] += inter
        batch_union_lst[i] += union
 
        ap = np.nan_to_num(get_ap_scores(output_AP, labels))
        f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
        batch_ap_lst[i] += ap
        batch_f1_lst[i] += f1

        bg_intersection_lst[i] = batch_inter_lst[i][0]
        fg_intersection_lst[i] = batch_inter_lst[i][1]
        tot_labeled_foreground_lst[i] = np.sum(labels[0].cpu().numpy() > 0)


    return batch_correct_lst, batch_label_lst, batch_inter_lst, batch_union_lst, batch_ap_lst, batch_f1_lst, bg_intersection_lst, fg_intersection_lst, tot_labeled_foreground_lst



res_dict = {}

res_dict[f'{args1.variant1}'] = {}
res_dict[f'{args1.variant2}'] = {}


current_args = args1
current_lrp_lst = variant1_lrp_lst
current_variant = args1.variant1

for k in range(2):


    if k==1:
        current_args = args2
        current_lrp_lst = variant2_lrp_lst
        current_variant = args2.variant


    print(current_variant)
    print(len(current_lrp_lst))
    for j in range(len(current_lrp_lst)):

        current_lrp = current_lrp_lst[j]
        epoch = current_args.custom_trained_models[j].split("_")[-1].split(".")[0]
        res_dict[f'{current_variant}'][epoch] = {}
        total_inter = [np.int64(0) for i in range(NUM_STEPS)]
        total_union = [np.int64(0) for i in range(NUM_STEPS)]
        total_correct = [np.int64(0) for i in range(NUM_STEPS)]
        total_label = [np.int64(0) for i in range(NUM_STEPS)]
 

        total_ap, total_f1 = [[] for i in range(NUM_STEPS)] , [[] for i in range(NUM_STEPS)]

        total_bg_intersection = [0 for i in range(NUM_STEPS)]
        total_fg_intersection = [0 for i in range(NUM_STEPS)]
        total_labeled_foreground = [0 for i in range(NUM_STEPS)]
        total_labeled_background = [0 for i in range(NUM_STEPS)]


        predictions, targets = [[] for i in range(NUM_STEPS)], [[] for i in range(NUM_STEPS)]


        pixAcc = [None for i in range(NUM_STEPS)]
        IoU  = [None for i in range(NUM_STEPS)]
        mIoU = [None for i in range(NUM_STEPS)]
        mAp = [None for i in range(NUM_STEPS)]
        mF1 = [None for i in range(NUM_STEPS)]
        mean_fg_intersection = [None for i in range(NUM_STEPS)]
        mean_bg_intersection = [None for i in range(NUM_STEPS)]

        count = 0

        for batch_idx, (image, labels) in enumerate(iterator):

            
        
            images = image.cuda()
            labels = labels.cuda()
            # print("image", image.shape)9
            # print("lables", labels.shape)

            correct_lst, labeled_lst, inter_lst, union_lst, ap_lst, f1_lst,  bg_intersection_lst, fg_intersection_lst, labeled_foreground_lst = eval_batch(images, labels, model, batch_idx)



            for i in range(NUM_STEPS):

                #predictions[i].append(pred_lst[i])
                #targets[i].append(target_lst[i])

                total_bg_intersection[i]+=bg_intersection_lst[i]
                total_fg_intersection[i]+=fg_intersection_lst[i]
                total_labeled_foreground[i]+=labeled_foreground_lst[i]
                total_labeled_background[i]+=(50176 - labeled_foreground_lst[i])

                total_correct[i] += correct_lst[i].astype('int64')
                total_label[i] += labeled_lst[i].astype('int64')
                total_inter[i] += inter_lst[i].astype('int64')
                total_union[i] += union_lst[i].astype('int64')
                total_ap[i] += [ap_lst[i]]
                total_f1[i] += [f1_lst[i]]
                pixAcc[i] = np.float64(1.0) * total_correct[i] / (np.spacing(1, dtype=np.float64) + total_label[i])
                IoU[i] = np.float64(1.0) * total_inter[i] / (np.spacing(1, dtype=np.float64) + total_union[i])
                mIoU[i] = IoU[i].mean()
                mAp[i] = np.mean(total_ap[i])
                mF1[i] = np.mean(total_f1[i])

                mean_fg_intersection[i] = total_fg_intersection[i] / total_labeled_foreground[i]
                mean_bg_intersection[i] = total_bg_intersection[i] / (total_labeled_background[i])
                iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc[i], mIoU[i], mAp[i], mF1[i]))
    
        res_dict[f'{current_variant}'][epoch]['mIoU'] = mIoU
        res_dict[f'{current_variant}'][epoch]['Pixel Accuracy'] = pixAcc
        res_dict[f'{current_variant}'][epoch]['mAP'] = mAp
        res_dict[f'{current_variant}'][epoch]['mean_bg_intersection'] = mean_bg_intersection
        res_dict[f'{current_variant}'][epoch]['mean_fg_intersection'] = mean_fg_intersection



print(res_dict)

update_json(f"testing/seg_results_compare.json", 
            res_dict)












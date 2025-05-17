import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from numpy import *
import argparse
from PIL import Image
import os
from tqdm import tqdm
from utils.metrices import *
from models.model_handler import model_env 
from utils.iou import IoU
import config
from data.imagenet_new import Imagenet_Segmentation
from misc.helper_functions import update_json
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2

from ViT_explanation_generator import Baselines, LRP
from old.model import deit_tiny_patch16_224 as vit_LRP
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F

plt.switch_backend('agg')


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
parser.add_argument('--custom-trained-model1', type=str, default="finetuned_models/IMNET/basic/checkpoint_0.pth",
                    help='Model path')
parser.add_argument('--threshold-type', choices = ['mean', 'otsu', 'MoV'], required = True)

parser.add_argument('--custom-trained-model2', type=str, default="finetuned_models/IMNET/basic/checkpoint_0.pth",
                    help='Model path')

parser.add_argument('--variant1', default = 'basic', help="")
parser.add_argument('--variant2', default = 'basic', help="")

parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
      
                    help='')
parser.add_argument('--mode', required= True, choices = ["check", "gen_images"])

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
parser.add_argument('--imagenet-seg-path',default="gtsegs_ijcv.mat", type=str, )
args1 = parser.parse_args()
args2 = parser.parse_args()


args1.variant = args1.variant1
args2.variant = args2.variant2


args1.custom_trained_model = args1.custom_trained_model1
args2.custom_trained_model = args2.custom_trained_model2


filenameJson = "testing/segmentation_debug_no_inter_results/fore_customLRP.json"
config.get_config(args1, skip_further_testing = True)
config.set_components_custom_lrp(args1)

config.get_config(args2, skip_further_testing = True)
config.set_components_custom_lrp(args2)

args1.checkname = args1.method + '_' + args1.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Define Saver
#saver = Saver(args)
#saver.results_dir = os.path.join(saver.experiment_dir, 'results')
#if not os.path.exists(saver.results_dir):
#    os.makedirs(saver.results_dir)
#if not os.path.exists(os.path.join(saver.results_dir, 'input')):
#    os.makedirs(os.path.join(saver.results_dir, 'input'))
#if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
#    os.makedirs(os.path.join(saver.results_dir, 'explain'))
#
#args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
#if not os.path.exists(args.exp_img_path):
#    os.makedirs(args.exp_img_path)
#args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
#if not os.path.exists(args.exp_np_path):
#    os.makedirs(args.exp_np_path)

# Data
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])



imagenet_trans = transforms.Compose([
        #transforms.Resize(sizeX, interpolation=3),
        #transforms.CenterCrop(args.input_size), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])





def concatenate_images_with_gaps(images, gap_size=10):
    """
    Concatenate multiple RGB images horizontally with gaps between them.
    
    Args:
        images: List of numpy arrays (RGB images)
        gap_size: Size of gap between images in pixels
    
    Returns:
        Combined RGB image as numpy array
    """
    # Get dimensions
    height = 224  # As specified in the requirements
    width = 224   # As specified in the requirements
    channels = 3  # RGB channels
    n_images = len(images)
    
    # Create empty array for the combined image with white background
    total_width = (n_images * width) + ((n_images - 1) * gap_size)
    combined_image = np.ones((height, total_width, channels), dtype=np.float32)
    
    # Place each image with gaps
    current_position = 0
    for img in images:
        # Ensure image is the correct size
        if img.shape != (height, width, channels):
            img = np.resize(img, (height, width, channels))
            
        # Place the image
        combined_image[:, current_position:current_position + width, :] = img
        current_position += width + gap_size
    
    return combined_image



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

if args1.mode == "gen_images":
    with open(filenameJson, 'r') as file:
        data = json.load(file)["res"]
    sorted_indices = [elem[0] for elem in data ]
    subset = Subset(ds, sorted_indices)
   
    dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

else:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

# Model

#FIXME: currently only attribution method is tested. Add support for other methods using other variants 
model = vit_LRP(pretrained=True).cuda()
baselines = Baselines(model)

if args1.custom_trained_model != None:
    if args1.data_set == 'IMNET100':
        args1.nb_classes = 100
    else:
        args1.nb_classes = 1000
      
    model_LRP1 = model_env(pretrained=False, 
                    args = args1,
                    hooks = True,
                )
    
    checkpoint = torch.load(args1.custom_trained_model, map_location='cpu')

    model_LRP1.load_state_dict(checkpoint['model'], strict=False)
    model_LRP1.to(device)


model_LRP1.eval()
lrp1 = LRP(model_LRP1)


if args2.custom_trained_model != None:
    if args2.data_set == 'IMNET100':
        args2.nb_classes = 100
    else:
        args2.nb_classes = 1000
      
    model_LRP2 = model_env(pretrained=False, 
                    args = args2,
                    hooks = True,
                )
    
    checkpoint = torch.load(args2.custom_trained_model, map_location='cpu')

    model_LRP2.load_state_dict(checkpoint['model'], strict=False)
    model_LRP2.to(device)


model_LRP2.eval()
lrp2 = LRP(model_LRP2)



# orig LRP
model_orig_LRP = vit_LRP(pretrained=True).cuda()
model_orig_LRP.eval()
orig_lrp = LRP(model_orig_LRP)

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

model.eval()


def concatenate_images_with_gaps(images, gap_size=10):
    """
    Concatenate multiple RGB images horizontally with gaps between them.
    
    Args:
        images: List of numpy arrays (RGB images)
        gap_size: Size of gap between images in pixels
    
    Returns:
        Combined RGB image as numpy array
    """
    # Get dimensions
    height = 224  # As specified in the requirements
    width = 224   # As specified in the requirements
    channels = 3  # RGB channels
    n_images = len(images)
    
    # Create empty array for the combined image with white background
    total_width = (n_images * width) + ((n_images - 1) * gap_size)
    combined_image = np.ones((height, total_width, channels), dtype=np.float32)
    
    # Place each image with gaps
    current_position = 0
    for img in images:
        # Ensure image is the correct size
        if img.shape != (height, width, channels):
            img = np.resize(img, (height, width, channels))
            
        # Place the image
        combined_image[:, current_position:current_position + width, :] = img
        current_position += width + gap_size
    
    return combined_image


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


def visualize(img, vis,idx):
    tensor_data1 = img.squeeze(0)  # Now it should be [3, 224, 224]
    image_data1 = np.transpose(tensor_data1, (1, 2, 0))  # Now it's [224, 224, 3]

    image_data1_Y =  (image_data1 - image_data1.min()) / (image_data1.max() - image_data1.min())

    image_data1_cc =  255 * (image_data1 - image_data1.min()) / (image_data1.max() - image_data1.min())
    image_data1_cc = image_data1_cc.astype('uint8')
    #image1 = Image.fromarray(image_data1_cc)
    #image_data1 = image_data1.reshape((224,224,3))

    tensor_data2 = vis  # Shape: [1, 3, 224, 224]




    x = np.uint8(255 * tensor_data2)

    heatmap = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(image_data1_Y)
    cam = cam / np.max(cam)

    cam =  np.uint8(255 * cam)
    cam = cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
    #plt.imsave(f"/content/{idx}.png"  ,image_data1_cc)
    #plt.imsave(f"/content/heat_{idx}.png"  ,cam)
    return cam, heatmap
    
    return image_data1_cc, heatmap
  


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()
    # Save input image


    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)
    
    # segmentation test for the rollout baseline
    if 'custom_lrp' in args1.method:
        Res1 = lrp1.generate_LRP(image.cuda(), method=f"{args1.method}",prop_rules = args1.prop_rules, cp_rule = args1.cp_rule).reshape(14, 14).unsqueeze(0).unsqueeze(0) 
        Res2 = lrp1.generate_LRP(image.cuda(), method=f"{args1.method}_SEMANTIC_ONLY",prop_rules = args1.prop_rules, cp_rule = args1.cp_rule).reshape(14, 14).unsqueeze(0).unsqueeze(0) 
        Res3 = lrp1.generate_LRP(image.cuda(), method=f"{args1.method}_PE_ONLY",prop_rules = args1.prop_rules, cp_rule = args1.cp_rule).reshape(14, 14).unsqueeze(0).unsqueeze(0) 
       

    elif args1.method == 'rollout':
        ResX = baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the LRP baseline (this is full LRP, not partial)

    elif 'full_lrp' in args1.method:
        Res1  = lrp1.generate_LRP(image.cuda(), method=args1.method,  cp_rule = args1.cp_rule, prop_rules = args1.prop_rules,  conv_prop_rule = args1.conv_prop_rule, index=None).reshape(1, 1, 224, 224).detach()
        Res2  = lrp2.generate_LRP(image.cuda(), method=args2.method,  cp_rule = args2.cp_rule, prop_rules = args2.prop_rules,  conv_prop_rule = args2.conv_prop_rule, index=None).reshape(1, 1, 224, 224).detach()
       
    # segmentation test for our method
    elif args1.method == 'transformer_attribution':
        Res1 = lrp1.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution", cp_rule = args1.cp_rule).reshape(batch_size, 1, 14, 14)
        Res2 = lrp2.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution", cp_rule = args2.cp_rule).reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the partial LRP baseline (last attn layer)
    elif args1.method == 'lrp_last_layer':
        ResX = orig_lrp.generate_LRP(image.cuda(), method="last_layer", is_ablation=args1.is_ablation, cp_rule = args1.cp_rule)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the raw attention baseline (last attn layer)
    elif args1.method == 'attn_last_layer':
        ResX = orig_lrp.generate_LRP(image.cuda(), method="last_layer_attn", is_ablation=args1.is_ablation, cp_rule = args1.cp_rule)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the GradCam baseline (last attn layer)
    elif args1.method == 'attn_gradcam':
        ResX = baselines.generate_cam_attn(image.cuda()).reshape(batch_size, 1, 14, 14)

    if 'full_lrp' not in args1.method:
        # interpolate to full image size (224,224)
        Res1 = torch.nn.functional.interpolate(Res1, scale_factor=16, mode='bilinear').cuda()
        Res2 = torch.nn.functional.interpolate(Res2, scale_factor=16, mode='bilinear').cuda()
        Res3 = torch.nn.functional.interpolate(Res3, scale_factor=16, mode='bilinear').cuda()


    
    # threshold between FG and BG is the mean    
    Res1 = (Res1 - Res1.min()) / (Res1.max() - Res1.min())
    Res2 = (Res2 - Res2.min()) / (Res2.max() - Res2.min())
    Res3 = (Res3 - Res3.min()) / (Res3.max() - Res3.min())



    if args1.threshold_type == 'otsu':
        ret1 = otsu_threshold(Res1.cpu().detach().numpy())
        ret2 =  otsu_threshold(Res2.cpu().detach().numpy())
        ret3 =  otsu_threshold(Res3.cpu().detach().numpy())


    else:
        ret1 = Res1.mean()
        ret2 = Res2.mean()
        ret3 = Res3.mean()


    Res1_1 = Res1.gt(ret1).type(Res1.type())
    Res1_0 = Res1.le(ret1).type(Res1.type())

    Res2_1 = Res2.gt(ret2).type(Res2.type())
    Res2_0 = Res2.le(ret2).type(Res2.type())

    Res3_1 = Res3.gt(ret3).type(Res3.type())
    Res3_0 = Res3.le(ret3).type(Res3.type())

    Res1_1_AP = Res1
    Res1_0_AP = 1-Res1

    Res2_1_AP = Res2
    Res2_0_AP = 1-Res2

    Res3_1_AP = Res3
    Res3_0_AP = 1-Res3

    Res1_1[Res1_1 != Res1_1] = 0
    Res1_0[Res1_0 != Res1_0] = 0
    Res1_1_AP[Res1_1_AP != Res1_1_AP] = 0
    Res1_0_AP[Res1_0_AP != Res1_0_AP] = 0

    Res2_1[Res2_1 != Res2_1] = 0
    Res2_0[Res2_0 != Res2_0] = 0
    Res2_1_AP[Res2_1_AP != Res2_1_AP] = 0
    Res2_0_AP[Res2_0_AP != Res2_0_AP] = 0


    Res3_1[Res3_1 != Res3_1] = 0
    Res3_0[Res3_0 != Res3_0] = 0
    Res3_1_AP[Res3_1_AP != Res3_1_AP] = 0
    Res3_0_AP[Res3_0_AP != Res3_0_AP] = 0

    # TEST
    pred1 = Res1.clamp(min=args1.thr) / Res1.max()
    pred1 = pred1.view(-1).data.cpu().numpy()
    
    pred2 = Res2.clamp(min=args1.thr) / Res2.max()
    pred2 = pred2.view(-1).data.cpu().numpy()

    pred3 = Res3.clamp(min=args1.thr) / Res3.max()
    pred3 = pred3.view(-1).data.cpu().numpy()

    target = labels.view(-1).data.cpu().numpy()
   

    output1 = torch.cat((Res1_0, Res1_1), 1)
    output_AP1 = torch.cat((Res1_0_AP, Res1_1_AP), 1)

    output2 = torch.cat((Res2_0, Res2_1), 1)
    output_AP2 = torch.cat((Res2_0_AP, Res2_1_AP), 1)

    output3 = torch.cat((Res3_0, Res3_1), 1)
    output_AP3 = torch.cat((Res3_0_AP, Res3_1_AP), 1)

    # Evaluate Segmentation
    batch_inter1, batch_union1, batch_correct1, batch_label1 = 0, 0, 0, 0
    batch_ap1, batch_f11 = 0, 0

    batch_inter2, batch_union2, batch_correct2, batch_label2 = 0, 0, 0, 0
    batch_ap2, batch_f12 = 0, 0

    batch_inter3, batch_union3, batch_correct3, batch_label3 = 0, 0, 0, 0
    batch_ap3, batch_f13 = 0, 0
    # Segmentation resutls
    correct1, labeled1 = batch_pix_accuracy(output1[0].data.cpu(), labels[0])
    inter1, union1 = batch_intersection_union(output1[0].data.cpu(), labels[0], 2)
    batch_correct1 += correct1
    batch_label1 += labeled1
    batch_inter1 += inter1
    batch_union1 += union1


    correct2, labeled2 = batch_pix_accuracy(output2[0].data.cpu(), labels[0])
    inter2, union2 = batch_intersection_union(output2[0].data.cpu(), labels[0], 2)
    batch_correct2 += correct2
    batch_label2 += labeled2
    batch_inter2 += inter2
    batch_union2 += union2

    correct3, labeled3 = batch_pix_accuracy(output3[0].data.cpu(), labels[0])
    inter3, union3 = batch_intersection_union(output3[0].data.cpu(), labels[0], 2)
    batch_correct3 += correct3
    batch_label3 += labeled3
    batch_inter3 += inter3
    batch_union3 += union3


    ap1 = np.nan_to_num(get_ap_scores(output_AP1, labels))
    f1_1 = np.nan_to_num(get_f1_scores(output1[0, 1].data.cpu(), labels[0]))
    batch_ap1 += ap1
    batch_f11 += f1_1


    ap2 = np.nan_to_num(get_ap_scores(output_AP2, labels))
    f1_2 = np.nan_to_num(get_f1_scores(output2[0, 1].data.cpu(), labels[0]))
    batch_ap2 += ap2
    batch_f12 += f1_2


    ap3 = np.nan_to_num(get_ap_scores(output_AP3, labels))
    f1_3 = np.nan_to_num(get_f1_scores(output3[0, 1].data.cpu(), labels[0]))
    batch_ap3 += ap3
    batch_f13 += f1_3


    pixAcc1 = batch_correct1 / batch_label1
    pixAcc2 = batch_correct2 / batch_label2
    pixAcc3 = batch_correct3 / batch_label3


    IOU1 = batch_inter1 / batch_union1
    IOU2 = batch_inter2 / batch_union2
    IOU3 = batch_inter3 / batch_union3



    mIOU1 = IOU1.mean()
    mIOU2 = IOU2.mean()
    mIOU3 = IOU3.mean()


    #Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
    #        os.path.join(saver.results_dir, '/content/The-Explainable-Transformer/testing/{}_mask.png'.format(index)))

    heatmap1,_ =  visualize(image.data.cpu().numpy(),Res1.reshape(224, 224).data.cpu().numpy(),index)
    heatmap2 ,_  =  visualize(image.data.cpu().numpy(),Res2.reshape(224, 224).data.cpu().numpy(),index)
    heatmap3 ,_  =  visualize(image.data.cpu().numpy(),Res3.reshape(224, 224).data.cpu().numpy(),index)

    #np.save(f"test_custom_{batch_idx}.npy", Res.reshape(224, 224).data.cpu().numpy() )
    label_img  = (labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8')
    #print(heatmap1)


    
    img = image[0].permute(1, 2, 0).data.cpu().numpy()
    img = 255 * (img - img.min()) / (img.max() - img.min())
    img = img.astype('uint8')



    Res1_mask = Res1_1.squeeze().data.cpu().numpy()
 
    Res1_mask = (np.stack([Res1_mask, Res1_mask, Res1_mask], axis=-1) * 255) .astype('uint8')


    Res2_mask = Res2_1.squeeze().data.cpu().numpy()
    Res2_mask = (np.stack([Res2_mask, Res2_mask, Res2_mask], axis=-1) * 255) .astype('uint8')


    Res3_mask = Res3_1.squeeze().data.cpu().numpy()
    Res3_mask = (np.stack([Res3_mask, Res3_mask, Res3_mask], axis=-1) * 255) .astype('uint8')
   
    #Res1_mask =( Res1.reshape(224, 224).data.cpu().numpy().repeat(3, 1, 1) * 255).astype('uint8')
    #Res2_mask =( Res2.reshape(224, 224).data.cpu().numpy().repeat(3, 1, 1) * 255).astype('uint8')


    #label_img its the gt mask
    #conc_imgs = concatenate_images_with_gaps([img, label_img, heatmap1, Res1_mask, heatmap2, Res2_mask ])
    #conc_imgs = conc_imgs.astype('uint8')
    #print(conc_imgs)
    #print(f"our pixelAccuracy: {pixAcc1}")
    #print(f"semantic pixelAccuracy: {pixAcc2}")
    #print(f"PE pixelAccuracy: {pixAcc3}")
#
    #print(f"our mIOU: {mIOU1}")
    #print(f"semantic mIOU: {mIOU2}")
    #print(f"PE mIOU: {mIOU3}")
#
#
    #print(f"basic num_correct: {batch_correct1}")
    #print(f"relu num_correct: {batch_correct2}")
#
    #print(f"basic IOU0: {batch_inter1[0]}") #batch_inter1[0] - intersection with background!!
    #print(f"relu IOU0: {batch_inter2[0]}")
    #print(f"basic IOU1: {batch_inter1[1]}") #intersection with foreground!!!!!
    #print(f"relu IOU1: {batch_inter2[1]}")


    #print(np.sum(labels[0].cpu().numpy() > 0))

    #print("\n")
    winner_pixAcc    = "ours" if pixAcc1 > pixAcc2 else "SEMANTIC"
    winner_miou      = "ours" if mIOU1 > mIOU2 else "SEMANTIC"
    winner_iou_fore  = "ours" if batch_inter1[1] > batch_inter2[1] else "SEMANTIC"  
    #winner_AP  = "ours" if batch_ap1 > batch_ap2 else "SEMANTIC"  



    if args1.mode == "gen_images":
        print(f"PIX: {pixAcc1} {pixAcc2} ")
        print(f"IOU: {mIOU1} {mIOU2} ")
        print(f"FIOU: {IOU1[0]} {IOU2[0]} ")
        print(f"AP: {batch_ap1} {batch_ap2} ")


        plt.imsave(f"testing/segmentation_debug_no_inter2/{index}.png", img)
        plt.imsave(f"testing/segmentation_debug_no_inter2/{index}_ours.png", heatmap1)
        plt.imsave(f"testing/segmentation_debug_no_inter2/{index}_SEMANTIC.png", heatmap2)
        plt.imsave(f"testing/segmentation_debug_no_inter2/{index}_POSITIONAL.png", heatmap3)
        conc_imgs = concatenate_images_with_gaps([img, label_img, heatmap1, Res1_mask, heatmap2, Res2_mask])
        conc_imgs = conc_imgs.astype('uint8')
        plt.imsave(f"testing/segmentation_debug_no_inter2/{index}_compare.png"  ,conc_imgs)


        return None

    if winner_iou_fore == "ours":
        print(f"FIOU: {batch_inter1[1]} {batch_inter2[1]} ")

        return [batch_idx, int(batch_inter1[1] - batch_inter2[1])]
        #return [batch_idx, batch_ap1[0] - batch_ap2[0]]
    
        
    return None


        #plt.imsave(f"testing/segmentation_debug/{index}_{winner_pixAcc}_{winner_miou}_{winner_iou_fore}.png"  ,conc_imgs)
    #plt.imsave(f"/content/heat_{index}.png"  ,heatmap1)
    #plt.imsave(f"/content/heat2_{index}.png"  ,heatmap2)
    #plt.imsave(f"/content/label_{index}.png"  ,label_img)



    return sorted_arr

os.makedirs("testing/segmentation_debug_no_inter2", exist_ok=True)
total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

predictions, targets = [], []
arr_orig = []
count = 0
for batch_idx, (image, labels) in enumerate(iterator):
    count+=1
    
    if count > 20 and args1.mode == "gen_images":
        break
    if args1.method == "blur":
        images = (image[0].cuda(), image[1].cuda())
    else:
        images = image.cuda()
    labels = labels.cuda()
    # print("image", image.shape)
    # print("lables", labels.shape)

    elem = eval_batch(images, labels, model, batch_idx)
    if elem !=None:
        arr_orig.append(elem)
    continue
    predictions.append(pred)
    targets.append(target)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_inter += inter.astype('int64')
    total_union += union.astype('int64')
    total_ap += [ap]
    total_f1 += [f1]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    #print(total_correct)
    #print(total_label)
    #print(pixAcc)
    IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
    
    print(total_union)
    print(total_inter)
    print(IoU)
    mIoU = IoU.mean()
    mAp = np.mean(total_ap)
    mF1 = np.mean(total_f1)
    iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))




if args1.mode == "gen_images":
    exit(1)


arr_orig = sorted(arr_orig, key=lambda x: x[1], reverse=True)
#print(arr_orig)

update_json(f"{filenameJson}", {f"res":arr_orig})


#predictions = np.concatenate(predictions)
#targets = np.concatenate(targets)
#pr, rc, thr = precision_recall_curve(targets, predictions)












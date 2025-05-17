#from models.model_wrapper import model_env 
#python check_conservation.py --auto --mode analyze_conservarion_per_image --method custom_lrp_gamma_rule_full

from ViT_explanation_generator import LRP, LRP_RAP
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm
import json
from torchvision import datasets
import argparse
from PIL import Image
import torch
from samples.CLS2IDX import CLS2IDX
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from misc.helper_functions import is_valid_directory ,create_directory_if_not_exists, update_json
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.model_visualizations import deit_tiny_patch16_224 as vit_LRP
from models.model_visualizations import deit_base_patch16_224 as vit_LRP_base
from models.model_visualizations import deit_small_patch16_224 as vit_LRP_small




import cv2
normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
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


def model_handler(pretrained=False,args  = None , hooks = False,  **kwargs):
    if "size" in args.model_components:
        if args.model_components['size'] == 'base':
            return vit_LRP_base(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    patch_embed          = args.model_components["patch_embed"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],


                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
            )
        elif args.model_components['size'] == 'small':
            return vit_LRP_small(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],

                    patch_embed          = args.model_components["patch_embed"],

                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
            )
    
    return vit_LRP(
            isWithBias           = args.model_components["isWithBias"],
            isConvWithBias       = args.model_components["isConvWithBias"],

            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],

            patch_embed          = args.model_components["patch_embed"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
   



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

def show_cam_on_image(img, mask):
   
    x = np.uint8(255 * mask)

    heatmap = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization_custom_LRP(original_image, class_index=None, method = None,  prop_rules = None, save_dir = None, posLens = False,  save_images_dir = None,batch_idx=None):
    res = []
    num_patch_above_thr = None
    if posLens:
        attributions = attribution_generator.generate_LRP(original_image.cuda(), prop_rules = prop_rules, method=method, cp_rule=args.cp_rule,  index=class_index)
        if save_images_dir == None:
            for i in range(original_image.shape[0]):
                res.append([])
                for elem in attributions:
                    res[-1].append(elem[i,:,:].detach().cpu().numpy().sum())

            attributions = res
    else:
        attributions = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), prop_rules = prop_rules, method=method, cp_rule=args.cp_rule,  index=class_index)
        attributions = [elem.detach().squeeze(0).squeeze(0).detach().cpu().numpy().sum() for elem in attributions]
        attributions = [float(elem) for elem in attributions]
      
    if save_images_dir:
       
        image_transformer_attribution = original_image[0].permute(1, 2, 0).data.cpu().numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
        
        
        image_copy = 255 *image_transformer_attribution
        image_copy = image_copy.astype('uint8')
        Image.fromarray(image_copy, 'RGB').save(f"{save_images_dir}/img_{batch_idx}.png")
       
        d = {0:"tot", 1:"sem", 2:"pos"}
       

        attributions = [elem[0,:] for elem in attributions]
        #print(attributions[0].shape)
        #print("here")
        for i in range(len(attributions)):
            attributions[i] = attributions[i].reshape(14, 14).unsqueeze(0).unsqueeze(0)
            attributions[i] = torch.nn.functional.interpolate(attributions[i], scale_factor=16, mode='bilinear', align_corners=False)
            attributions[i] = attributions[i].squeeze().detach().cpu().numpy()
            attributions[i] = (attributions[i] - attributions[i].min()) / (attributions[i].max() - attributions[i].min())



            vis = show_cam_on_image(image_transformer_attribution, attributions[i])
            vis =  np.uint8(255 * vis)
            vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            plt.imsave(f"{save_images_dir}/img_{batch_idx}_{d[i]}.png" , vis)

       
    bin_mask0 =  otsu_threshold(attributions[0])
    bin_mask1 =  otsu_threshold(attributions[1])
    bin_mask2 =  otsu_threshold(attributions[2])

    bw0 = np.where(attributions[0] > bin_mask0, 1, 0)
    bw1 = np.where(attributions[1] > bin_mask0, 1, 0)
    bw2 = np.where(attributions[2] > bin_mask0, 1, 0)

    bw0 = (np.stack([bw0, bw0, bw0], axis=-1) * 255) .astype('uint8')
    bw1 = (np.stack([bw1, bw1, bw1], axis=-1) * 255) .astype('uint8')
    bw2 = (np.stack([bw2, bw2, bw2], axis=-1) * 255) .astype('uint8')


    conc_imgs = concatenate_images_with_gaps([image_copy, bw0, bw1, bw2])
    conc_imgs = conc_imgs.astype('uint8')
    plt.imsave(f"{save_images_dir}/img_{batch_idx}_compare.png" , conc_imgs)
 

    num_patch_above_thr = ((attributions[0] > bin_mask0).sum()) / (224*224)
 
    return attributions, num_patch_above_thr



import random

def swap_blocks_in_image(x,grid_size, inner_size=192, return_full_size=True):
  
    # Original dimensions
    _, _, h, w = x.shape
    
    # Step 1: Extract the inner region
    start_idx = (h - inner_size) // 2
    end_idx = start_idx + inner_size
    inner_region = x[:, :, start_idx:end_idx, start_idx:end_idx].clone()  # Shape: [1, 3, 192, 192]
   
    # Step 2: Divide into 3×3 blocks
    block_size = inner_size // grid_size  # Each block is 64×64
    shuffled_inner = torch.zeros_like(inner_region)

    # Sample 2 pairs of blocks to swap (4 unique blocks)
    block_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    shuffled_positions = block_positions.copy()
    random.shuffle(shuffled_positions)

    for idx, ((orig_y, orig_x), (new_y, new_x)) in enumerate(zip(block_positions, shuffled_positions)):
        # Calculate original and new pixel coordinates
        orig_y_pixel = orig_y * block_size
        orig_x_pixel = orig_x * block_size
        new_y_pixel = new_y * block_size
        new_x_pixel = new_x * block_size
        
        # Move the block to its new position
        shuffled_inner[:, :, 
                      new_y_pixel:new_y_pixel+block_size, 
                      new_x_pixel:new_x_pixel+block_size] = inner_region[:, :, 
                                                                        orig_y_pixel:orig_y_pixel+block_size, 
                                                                        orig_x_pixel:orig_x_pixel+block_size].clone()
    
    # Step 4: Return either inner region or full-sized tensor
    if return_full_size:
        result = x.clone()
        result[:, :, start_idx:end_idx, start_idx:end_idx] = shuffled_inner
        return result
    else:
        return shuffled_inner

def analyze_conservation_per_image(model, args=None):
    d = {}
    res = []
    save_file = f"testing/visualizations_conservation/basic/{args.method}/acc_results_all_imgs.json"
    with open(save_file, 'r') as file:
        data = json.load(file)
       
    
    for k in range(len(data["total"])):
        ratio = data["pe"][k] /  data["total"][k]
        res.append([f"{k}", ratio])
    
    best_res  = [elem for elem in res]
    worst_res = [elem for elem in res]

    best_res.sort(key=lambda x: x[1],reverse=True)

    best_res = best_res[:1000]

    best_idx = [int(elem[0]) for elem in best_res]

    worst_res.sort(key=lambda x: x[1])
    worst_res = worst_res[:1000]
    worst_idx = [int(elem[0]) for elem in worst_res]

   
    best_dir = f"testing/visualizations_conservation/basic/{args.method}/best_imgs"
    worst_dir = f"testing/visualizations_conservation/basic/{args.method}/worst_imgs"

    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir , exist_ok=True)
  
    indices = worst_idx + best_idx
    root = os.path.join(args.data_path,  'val')
    dataset_val = datasets.ImageFolder(root, transform=transform)
    subset = Subset(dataset_val, indices)

    sample_loader = torch.utils.data.DataLoader(    
    subset, 
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last = True
    )

    avg_acc_best  = 0.0
    avg_acc_worst = 0.0
    avg_acc_best_shuffle  = 0.0
    avg_acc_worst_shuffle  = 0.0

    total_best  = 0
    total_worst = 0

    top1_hit_best  = 0
    top1_hit_worst = 0
    top5_hit_best  = 0
    top5_hit_worst = 0
    num_patch_above_best= 0
    num_patch_above_worst= 0


    d_best = {}
    d_worst = {}

    d_dist_best = {}
    d_dist_worst = {}
    thr = 500
    


    for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
      isBest = False
      if (batch_idx>=1000):
          isBest = True
          
      elif (batch_idx<1000):
          pass
      else:
          #print("HERE")
          
          continue
 
     
      save_images_dir = best_dir if isBest else worst_dir
      output = model(data.cuda()).detach()
      topk_values, _ = torch.topk(output, k=2, dim=1, largest=True, sorted=True)
   


      max_logits, max_indices = torch.max(output, dim=1)
    
   
      if (max_indices.item() != target.item()):
          continue
      
      if ((isBest and total_best == thr) or ((isBest==False)and total_worst == thr)):
          continue
    
      class_name = CLS2IDX[target.item()]
      if isBest:
          total_best+=1
          avg_acc_best+=max_logits
          if class_name not in d_best:
              d_best[class_name]  =0
          d_best[class_name]+=1
          d_dist_best[f"img{batch_idx} | {class_name}"] = topk_values[0][0] - topk_values[0][1]
      else:
          total_worst+=1
          avg_acc_worst+=max_logits

          if class_name not in d_worst:
              d_worst[class_name]  =0
          d_worst[class_name]+=1
          d_dist_worst[f"img{batch_idx} | {class_name}"] = topk_values[0][0] - topk_values[0][1]


      data = swap_blocks_in_image(data,16) #4
      output = model(data.cuda()).detach()
      max_logits, max_indices = torch.max(output, dim=1)

      topk_values, topk_indices = torch.topk(output, k=5, dim=1, largest=True, sorted=True)
      logits_class = output[0,target.item()]


      if isBest:
        
        if (topk_indices[0][0].item()==target.item()):
            top1_hit_best+=1

        if (topk_indices==target.item()).any():
            top5_hit_best+=1
        avg_acc_best_shuffle+=logits_class
      else:
        if (topk_indices[0][0].item()==target.item()):
            top1_hit_worst+=1
        if (topk_indices==target.item()).any():
             top5_hit_worst+=1
        avg_acc_worst_shuffle+=logits_class
      ''' 
      res, num_patch_above_thr = generate_visualization_custom_LRP(data , 
                                             None,
                                             prop_rules = args.prop_rules,
                                             method = args.method,
                                             save_dir = best_dir,
                                             posLens = True ,
                                            save_images_dir = save_images_dir,
                                            batch_idx =f"{batch_idx}_{CLS2IDX[target.item()]}" 
                                            )
      
      if isBest:
        num_patch_above_best+=num_patch_above_thr
      else:
        num_patch_above_worst+=num_patch_above_thr

     '''
    
    print(f"BEST: total {total_best}")
    print(f"WORST: total {total_worst}")
    print(f"BEST: avg logit before {avg_acc_best/ total_best}")
    print(f"WORST: avg logit before {avg_acc_worst/ total_worst}")
    print(f"BEST SHUFFLE: avg logit after {avg_acc_best_shuffle/ total_best}")
    print(f"WORST SHUFFLE: avg logit after {avg_acc_worst_shuffle/ total_worst}")
    print("\n\n")
    print(f"BEST SHUFFLE: top1  {top1_hit_best/ total_best}")
    print(f"WORST SHUFFLE: top1 {top1_hit_worst/ total_worst}")
    print(f"BEST SHUFFLE: top5  {top5_hit_best/ total_best}")
    print(f"WORST SHUFFLE: top5 {top5_hit_worst/ total_worst}")
    print(f"BEST  above thr:  {num_patch_above_best/ total_best}")
    print(f"WORST above thr: {num_patch_above_worst/ total_worst}")

    count = 0
    print("BEST CLASSES")
    for key in sorted(d_best, key=d_best.get, reverse=True):
        count+=1
        if count == 5:
            break
        print(f"{count}: {key} {d_best[key]}")
    count = 0
    print("WORST CLASSES")
    for key in sorted(d_worst, key=d_worst.get, reverse=True):
        count+=1
        if count == 4:
            break
        print(f"{count}: {key} {d_worst[key]}")

    print("------")
    count = 0
    for key in sorted(d_dist_best, key=d_dist_best.get, reverse=True):
        count+=1
        if count == 20:
            break
        print(f"{count}: {key} {d_dist_best[key]}")
    print("\n\n")

    count = 0
    for key in sorted(d_dist_worst, key=d_dist_worst.get, reverse=True):
        count+=1
        if count == 20:
            break
        print(f"{count}: {key} {d_dist_worst[key]}")
    

def analyze_posLens(save_dir = "testing/visualizations_conservation"):
    d = {}
    res = []
    for dir in os.listdir(save_dir):
        if  dir != "base_small":
            continue
        if os.path.isdir(f"{save_dir}/{dir}") == False:
            continue
        save_file = f"{save_dir}/{dir}/custom_lrp_gamma_rule_full/posLens_results.json" 
        with open(save_file, 'r') as file:
            data = json.load(file)
        
        for k in data:

            ratio = data[k]["PE"] /  data[k]["ours"]
            res.append([f"{k}", ratio])
    res.sort(key=lambda x: x[1],reverse=True)
    res = res[:10]
    top_classes = [int(elem[0]) for elem in res]
 

    update_json(f'testing/visualizations_conservation/base_small/custom_lrp_gamma_rule_full/top.json',{"top":top_classes} )

   
    names = [item[0] for item in res]
    values = [item[1] for item in res]
    plt.bar(names, values)
    plt.savefig(f"{save_dir}/pos_lens.png", dpi=300, bbox_inches='tight')

    exit(1)
    positions = []
    group_positions = [] 
   
    for i in range(len(res)):
        group_idx = i // 3
        within_idx = i % 3
        # Each group starts at position group_idx * 4
        # Within a group, bars are at positions 0, 1, 2
        # This creates a gap of 1 unit between groups
        positions.append(group_idx * 4 + within_idx)
        if within_idx == 1:
            group_positions.append(positions[-1])
       
    
   
    fig, ax = plt.subplots(figsize=(28, 22))
    group_labels = ["Ours", "AttnLRP", "PE Only"]
    bar_labels = group_labels * 3
  
   
    ax.set_xticks(positions, bar_labels, rotation=45, ha='right', fontsize=48)
    #for i, pos in enumerate(group_positions):
    #    ax.text(pos, 2.0, group_labels[i], ha='center', fontsize=51, fontweight='bold')
    
    ax.bar(positions[0:3], res[0:3], label='Tiny',zorder=2)
    ax.bar(positions[3:6], res[3:6], label='Small',zorder=2)
    ax.bar(positions[6:], res[6:], label='Medium',zorder=2)

    ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)
    ax.tick_params(axis='y', labelsize=58)
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, markerscale=1, fontsize=45, frameon=False)

    plt.savefig(f"{save_dir}/conservation_results.png", dpi=300, bbox_inches='tight')




def analyze(save_dir = "testing/visualizations_conservation"):
    d = {}
    res = []
    for dir in os.listdir(save_dir):
        if os.path.isdir(f"{save_dir}/{dir}") == False:
            continue
        save_file = f"{save_dir}/{dir}/custom_lrp_gamma_rule_full/acc_results.json" 
        with open(save_file, 'r') as file:
            data = json.load(file)
        total = sum(data["total"] )/ len( data["total"])
        semantic = sum(data["semantic"] )/ len( data["semantic"])
        pe = sum(data["pe"] )/ len( data["pe"])
        d[dir] = [total,semantic,pe]
    
    
    res+= d["basic"]
    res+= d["base_small"]
    res+= d["basic_medium"]

    print(f'tiny: {d["basic"][2] / d["basic"][0]}')
    print(f'small: {d["base_small"][2] / d["base_small"][0]}')
    print(f'medium: {d["basic_medium"][2] / d["basic_medium"][0]}')

    
    positions = []
    group_positions = [] 
   
    for i in range(len(res)):
        group_idx = i // 3
        within_idx = i % 3
        # Each group starts at position group_idx * 4
        # Within a group, bars are at positions 0, 1, 2
        # This creates a gap of 1 unit between groups
        positions.append(group_idx * 4 + within_idx)
        if within_idx == 1:
            group_positions.append(positions[-1])
       
    
   
    fig, ax = plt.subplots(figsize=(28, 22))
    group_labels = ["Ours", "AttnLRP", "PE Only"]
    bar_labels = group_labels * 3
  
   
    ax.set_xticks(positions, bar_labels, rotation=45, ha='right', fontsize=48)
    #for i, pos in enumerate(group_positions):
    #    ax.text(pos, 2.0, group_labels[i], ha='center', fontsize=51, fontweight='bold')
    
    ax.bar(positions[0:3], res[0:3], label='Tiny',zorder=2)
    ax.bar(positions[3:6], res[3:6], label='Small',zorder=2)
    ax.bar(positions[6:], res[6:], label='Medium',zorder=2)

    ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)
    ax.tick_params(axis='y', labelsize=58)
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, markerscale=1, fontsize=45, frameon=False)

    plt.savefig(f"{save_dir}/conservation_results.png", dpi=300, bbox_inches='tight')




if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  
  parser.add_argument('--custom-trained-model', 
                     
                        help='')
  parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

  parser.add_argument('--data-path', )
  
  parser.add_argument('--variant', default = 'basic' , type=str, help="")
  parser.add_argument('--mode', required= True, choices = ["check_conservarion", "check_conservarion_per_image","check_posLens", "analyze_conservarion", "analyze_conservarion_per_image", "analyze_posLens"])

  parser.add_argument('--class-index', 
                       # default = "243",
                       type=int,
                        help='') #243 - dog , 282 - cat
  parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        help='')
  
  parser.add_argument('--auto', action='store_true')
  
  parser.add_argument('--grid', action='store_true')
  parser.add_argument('--save_images', action='store_true')


  
  args = parser.parse_args()

  save_dir = f"testing/visualizations_conservation/{args.variant}/{args.method}"
  imgs_save_dir = f"testing/visualizations_conservation/{args.variant}/{args.method}/images"
  top_file    = f"testing/visualizations_conservation/{args.variant}/{args.method}/top.json"
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(imgs_save_dir , exist_ok=True)

  if args.save_images:
      with open(top_file, 'r') as file:
        top_data = json.load(file)
      top_classes = top_data["top"][:10]
      top_classes.append(1)
      for c in top_classes:
          os.makedirs(f"{imgs_save_dir}/{c}" , exist_ok=True)
          

  if args.mode == "analyze_conservarion":
     analyze()
     exit(1)
  if args.mode == "analyze_posLens":
     analyze_posLens()
     exit(1) 

  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
  config.set_components_custom_lrp(args,gridSearch = args.grid)
  if args.auto:
    exp_name      = args.variant
    exp_name = f'{args.data_set}/{exp_name}' 
    results_exp_dir              = f'{args.dirs["results_dir"]}/{exp_name}'
    args.custom_trained_model    =  is_valid_directory(results_exp_dir)
  

 

  
  if args.data_set == "IMNET100":
    args.nb_classes = 100
  else:
     args.nb_classes = 1000

  
  model = model_handler(pretrained=False, 
                      args = args,
                      hooks = True,
                    )
        #model_LRP.head = torch.nn.Linear(model_LRP.head.weight.shape[1],100)
  
  
  checkpoint = torch.load(args.custom_trained_model, map_location='cpu')
  print(torch.cuda.is_available())
  print(torch.__version__)
  model.load_state_dict(checkpoint['model'], strict=False)
  model.cuda()
  model.eval()
  attribution_generator = LRP(model)

  if args.mode == "analyze_conservarion_per_image":
     analyze_conservation_per_image(model, args=args)
     exit(1) 

  if args.mode == "check_conservarion":
    dataset_val = datasets.ImageFolder("val", transform=transform)
    np.random.seed(42)
    torch.manual_seed(42)
    total_size  = len(dataset_val)
    indices = list(range(total_size))
    subset_size = int(total_size *0.2)
    random_indices = np.random.choice(indices, size=subset_size, replace=False)
    sampler = SubsetRandomSampler(random_indices)  
   
    sample_loader = torch.utils.data.DataLoader(    
        dataset_val, sampler=sampler,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last = True  
    )
  else:
        num_workers = 2 if args.mode == "check_posLens" else 1
        batch_size = 16 if args.mode == "check_posLens" else 1

        root = os.path.join(args.data_path,  'val')
        dataset_val = datasets.ImageFolder(root, transform=transform)
        sample_loader = torch.utils.data.DataLoader(    
        dataset_val, 
        batch_size=batch_size if args.save_images == False else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last = True
    )

  total = []
  semantic = []
  pe = []
  hist = np.array([{'ours':0,'SEM':0,'PE':0, 'count':0} for i in range(1000)])

  for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
    #print(data.shape)
    #print(target.shape)
    #exit(1)
    
    if args.save_images:
        print(target.item())
        if target.item() not in top_classes:
            continue
    
    data  = data.squeeze(0) if "check_conservarion" in args.mode else data
    

    res, _ = generate_visualization_custom_LRP(data , 
                                             None,
                                             prop_rules = args.prop_rules,
                                             method = args.method,
                                             save_dir = save_dir,
                                             posLens = "check_conservarion" not in args.mode ,
                                            save_images_dir = f"{imgs_save_dir}/{target.item()}" if args.save_images else None,
                                            batch_idx = batch_idx
                                            )
   
    if "check_conservarion" in args.mode:
        total.append(res[0])
        semantic.append(res[1])
        pe.append(res[2])
    else:
        for idx, c in enumerate(target):
            hist[c]['ours']+=res[idx][0]
            hist[c]['SEM']+=res[idx][1]
            hist[c]['PE']+=res[idx][2]
            hist[c]['count']+=1


filename = None
if args.mode == "check_conservarion":
    filename = 'acc_results.json'
elif args.mode == "check_conservarion_per_image":
    filename = 'acc_results_all_imgs.json'
else:
    filename =  'posLens_results.json'
if "check_conservarion" in args.mode:

    update_json(f'{save_dir}/{filename}', {"total":total, "semantic": semantic, "pe": pe})
else:
    if args.save_images == False:

        update_json(f'{save_dir}/{filename}', {f"{i}":hist[i] for i in range(1000)})



  #saved_image_path = f"{save_dir}/{img_name}_{args.method}.png"
  

 
  #plt.imsave(saved_image_path, vis)

  



#print_top_classes(output)









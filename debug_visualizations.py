#from models.model_wrapper import model_env 
from models.model_handler import model_env 
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
from ViT_explanation_generator import LRP, LRP_RAP
import torchvision.transforms as transforms
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


import cv2
normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])




# create heatmap from mask on image
def show_cam_on_image(img, mask):
   
    x = np.uint8(255 * mask)

    heatmap = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization_transformer_attribution(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
  
   
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



def generate_visualization_full_LRP(original_image, class_index=None, method = None, prop_rules= None, conv_prop_rule = None, i=0):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method=method,  cp_rule=args.cp_rule, prop_rules = prop_rules,  index=class_index, conv_prop_rule = conv_prop_rule).detach()
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(image_transformer_attribution.shape)
    image_copy = 255 *image_transformer_attribution
    image_copy = image_copy.astype('uint8')
    Image.fromarray(image_copy, 'RGB').save(f'testing/visualizations_view/img_{i}.png')
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def generate_visualization_RAP(original_image, class_index=None, epsilon_rule = False, gamma_rule = False, default_op=None):
    transformer_attribution = RAP_generator.generate_LRP_RAP(original_image.unsqueeze(0).cuda(), method="full",  cp_rule=args.cp_rule,  index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(image_transformer_attribution.shape)
    
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def generate_visualization_custom_LRP(original_image, class_index=None, method = None, prop_rules = None, i = None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), prop_rules = prop_rules, method=method, cp_rule=args.cp_rule,  index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(14, 14).unsqueeze(0).unsqueeze(0)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear', align_corners=False)
    transformer_attribution = transformer_attribution.squeeze().detach().cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    #transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    #transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(transformer_attribution)

    image_copy = 255 *image_transformer_attribution
    image_copy = image_copy.astype('uint8')
    Image.fromarray(image_copy, 'RGB').save(f'testing/visualizations_view/img_{i}.png')
   
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def create_image_pdf(input_dir="testing/visualizations_view", output_pdf= "testing/showcase.pdf", images_per_page=5):
    """
    Create a PDF with normal images and their corresponding heatmaps arranged in columns.

    Args:
        input_dir (str): Directory containing the images and heatmaps
        output_pdf (str): Output PDF file path
        images_per_page (int): Number of images per page
    """
    # Get all files in directory
    files = os.listdir(input_dir)

    # Separate normal images and heatmaps
    normal_images = sorted([f for f in files if f.startswith('img_')])
    heatmaps = [f for f in files if not f.startswith('img_')]

    # Group heatmaps by their base number
    heatmap_groups = {}
    for hm in heatmaps:
        num = re.match(r'(\d+)_', hm)
        if num:
            num = num.group(1)
            if num not in heatmap_groups:
                heatmap_groups[num] = []
            heatmap_groups[num].append(hm)

    # Sort heatmaps within each group
    for num in heatmap_groups:
        heatmap_groups[num].sort()
        #print(heatmap_groups[num])
        #exit(1)
    # Calculate dimensions
    page_width, page_height = letter
    margin = 50
    usable_width = page_width - 2 * margin
    usable_height = page_height - 2 * margin

    # Determine number of columns (1 for normal image + number of variants)
    max_variants = max(len(group) for group in heatmap_groups.values()) if heatmap_groups else 0
    num_columns = 1 + max_variants

    # Calculate image dimensions
    image_width = (usable_width - (num_columns - 1) * 20) / num_columns  # 20px spacing between columns
    image_height = (usable_height - (images_per_page - 1) * 20) / images_per_page  # 20px spacing between rows

    # Create PDF
    c = canvas.Canvas(output_pdf, pagesize=letter)
    normal_images = sorted(normal_images, key=lambda x: int(re.search(r'(\d+)', x).group(0)))

    for i, img_file in enumerate(normal_images):
        #if i==4:
        #  break
        # Start new page if needed
        if i % images_per_page == 0 and i != 0:
            c.showPage()

        # Calculate row position (from bottom, since ReportLab uses bottom-left as origin)
        row = i % images_per_page
        y_position = page_height - margin - row * (image_height + 20) - image_height

        # Draw normal image
        img_path = os.path.join(input_dir, img_file)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Resize image maintaining aspect ratio
            img.thumbnail((image_width, image_height))
            c.drawImage(img_path, margin, y_position, width=image_width, height=image_height, preserveAspectRatio=True)

        # Draw corresponding heatmaps
        num = re.match(r'img_(\d+)\.png', img_file).group(1)
        if num in heatmap_groups:
            for j, heatmap_file in enumerate(heatmap_groups[num]):
                hm_path = os.path.join(input_dir, heatmap_file)
                if os.path.exists(hm_path):
                    x_position = margin + (j + 1) * (image_width + 20)
                    c.drawImage(hm_path, x_position, y_position, width=image_width, height=image_height, preserveAspectRatio=True)

    # Save PDF
    c.save()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()

 
  parser.add_argument('--idx', 
                        type=int,
                        help='')
  parser.add_argument('--custom-trained-model', 
                     
                        help='')
  parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

  parser.add_argument('--data-path', default='',)
  
  parser.add_argument('--variant', default = 'basic' , type=str, help="")
  parser.add_argument('--class-index', 
                       # default = "243",
                       type=int,
                        help='') #243 - dog , 282 - cat
  parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        help='')

  parser.add_argument('--collect', action='store_true')

  parser.add_argument('--auto', action='store_true')
  parser.add_argument('--grid', action='store_true')


  args = parser.parse_args()
  if args.collect:
     create_image_pdf()
     exit(1)

  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
  config.set_components_custom_lrp(args,gridSearch = args.grid)
  if args.auto:
    exp_name      = args.variant
    exp_name = f'{args.data_set}/{exp_name}' 
    results_exp_dir              = f'{args.dirs["results_dir"]}/{exp_name}'
    args.custom_trained_model    =  is_valid_directory(results_exp_dir)
  

  save_dir = f"testing/visualizations_view"
  os.makedirs(f"testing/visualizations_view", exist_ok=True)

 
  if args.data_set == "IMNET100":
    args.nb_classes = 100
  else:
     args.nb_classes = 1000

  
  model = model_env(pretrained=False, 
                      args = args,
                      hooks = True,
                    )
        #model_LRP.head = torch.nn.Linear(model_LRP.head.weight.shape[1],100)
  checkpoint = torch.load(args.custom_trained_model, map_location='cpu')

  model.load_state_dict(checkpoint['model'], strict=False)
  model.cuda()
  model.eval()
  attribution_generator = LRP(model)
  RAP_generator = LRP_RAP(model)


  dataset_val = datasets.ImageFolder("val", transform=transform)
  np.random.seed(42)
  torch.manual_seed(42)
  total_size  = len(dataset_val)
  indices = list(range(total_size))
  subset_size = int(total_size *0.04)
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

  for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):

    #if batch_idx > 3:
    #   break

    if "custom_RAP" in args.method:
      vis = generate_visualization_RAP(data, args.class_index, epsilon_rule = epsilon_rule, gamma_rule = gamma_rule, default_op = default_op)
      method_name = "custom_RAP"
    elif args.method == "transformer_attribution" or args.method == "attribution_with_detach":
      vis = generate_visualization_transformer_attribution(data, args.class_index)
      method_name = "Att"
    elif "custom_lrp" in args.method:

      vis = generate_visualization_custom_LRP(data.squeeze(0), 
                                              args.class_index,
                                             prop_rules = args.prop_rules,
                                             method = args.method,
                                             i=batch_idx

                                              )
      method_name = "custom_lrp"
    else:
      conv_prop_rule = args.conv_prop_rule
      vis = generate_visualization_full_LRP(data.squeeze(0), 
                                            args.class_index,
                                             method = args.method,
                                            prop_rules = args.prop_rules,
                                            conv_prop_rule = conv_prop_rule,
                                            i=batch_idx
                                              )




    saved_image_path = f"{save_dir}/{batch_idx}_{args.variant}_{args.method}.png"
 
    plt.imsave(saved_image_path, vis)

  



#print_top_classes(output)









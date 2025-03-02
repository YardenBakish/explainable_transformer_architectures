#from models.model_wrapper import model_env
from models.model_handler import model_env

from ViT_explanation_generator import LRP
import torchvision.transforms as transforms
import argparse
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from misc.helper_functions import *

from tqdm import tqdm

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import cv2
normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

imagenet_normalize = transforms.Compose([
    #transforms.Resize(256, interpolation=3),
    #transforms.CenterCrop(224),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])


def compute_SaCo(mapper,img_idx,num_steps):
    """
    Salience-guided Faithfulness Coefficient (SaCo).

    Parameters:
    - model: Pre-trained model.
    - explanation_method: Explanation method.
    - x: Input image.
    - K: Number of groups G.

    Returns:
    - F: Faithfulness coefficient.
    """

    # initialize F and totalWeight
    F = 0.0
    totalWeight = 0.0


    # Generate Gi based on the saliency map M(x, ŷ) and K
    # Then compute the corresponding s(Gi) and ∇pred(x, Gi) for i = 1, 2, ..., K
    s_G = mapper[img_idx]['sum_salience']  # the list of s(Gi)
    pred_x_G = mapper[img_idx]['perturbated_predicitions_diff']  # the list of ∇pred(x, Gi)

    # Traverse all combinations of Gi and Gj
    for i in range(num_steps - 1):
        for j in range(i + 1, num_steps):
            if pred_x_G[i] >= pred_x_G[j]:
                weight = s_G[i] - s_G[j]
            else:
                weight = -(s_G[i] - s_G[j])

            F += weight
            totalWeight += abs(weight)

    if totalWeight != 0:
        F /= totalWeight
    else:
        raise ValueError("The total weight is zero.")

    return F



def aggrage_info(batch_idx=0,  original_image= None, mapper = None, args = None):
    epsilon_rule = args.epsilon_rule
    gamma_rule   = args.gamma_rule
    default_op   = args.default_op
    num_steps    = args.num_steps
    #conv_prop_rule = args.conv_prop_rule
    mapper[batch_idx] = {}
    perturbation_steps = [i/num_steps for i in range(num_steps+1)]
    image_transformed = imagenet_normalize(original_image)
    output = model(image_transformed.unsqueeze(0).cuda()).detach()
    class_pred_idx = output.data.topk(1, dim=1)[1][0].tolist()[0]
    prob = torch.softmax(output, dim=1)
    pred_prob = prob[0,class_pred_idx]



    mapper[batch_idx]['perturbated_predicitions_diff'] = []
    mapper[batch_idx]['sum_salience']                  =  []

    if "full_lrp" in  args.method:
     transformer_attribution = attribution_generator.generate_LRP(image_transformed.unsqueeze(0).cuda(),  method="full_lrp", cp_rule=args.cp_rule, epsilon_rule = epsilon_rule, gamma_rule=gamma_rule, default_op=default_op, conv_prop_rule = args.conv_prop_rule ).detach()
    else:
        transformer_attribution = attribution_generator.generate_LRP(image_transformed.unsqueeze(0).cuda(),  method="custom_lrp", cp_rule=args.cp_rule, epsilon_rule = epsilon_rule, gamma_rule=gamma_rule, default_op=default_op ).detach()
        transformer_attribution = transformer_attribution.reshape(14, 14).unsqueeze(0).unsqueeze(0)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear', align_corners=False)
    transformer_attribution = transformer_attribution.squeeze()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    transformer_attribution_copy = transformer_attribution.detach().cpu().numpy()


    image_copy = original_image.clone()
    image_NP = image_copy.permute(1, 2, 0).detach().cpu().numpy()

    image_copy_tensor = image_copy.cuda()

    org_shape = image_copy.shape
    transformer_attribution   = transformer_attribution.reshape(1, -1)
    base_size = 224 * 224


    mean_pixel0 = None

    for i in range(1, len(perturbation_steps)):
      _data_pred_pertubarted           = image_copy_tensor.clone()
      _, idx_pred   = torch.topk(transformer_attribution, int(base_size * perturbation_steps[i]), dim=-1)
      idx_pred = idx_pred[:,int(base_size * perturbation_steps[i-1]):]

      mapper[batch_idx]['sum_salience'].append(transformer_attribution.flatten()[idx_pred].sum().item())
      idx_pred = idx_pred.squeeze()  # Remove any extra dimensions
      idx_pred = idx_pred.unsqueeze(0)  # Add channel dimension
      idx_pred = idx_pred.expand(3, -1)


      _data_pred_pertubarted                 = _data_pred_pertubarted.reshape(3, -1)
      if i == 1:
        mean_pixel0 = _data_pred_pertubarted.mean(dim=1,keepdim = True)



      mean_pixel = mean_pixel0.expand(-1, idx_pred.size(1))
      _data_pred_pertubarted                 = _data_pred_pertubarted.scatter_(1, idx_pred, mean_pixel)
      _data_pred_pertubarted_image           = _data_pred_pertubarted.clone().reshape(3, 224, 224).permute(1, 2, 0)

      res =  (_data_pred_pertubarted_image.cpu().numpy() * 255).astype('uint8')
      _data_pred_pertubarted               = _data_pred_pertubarted.reshape(*org_shape)
      _norm_data_pred_pertubarted            = normalize(_data_pred_pertubarted.unsqueeze(0))

      out_data_pred_pertubarted              = model(_norm_data_pred_pertubarted).detach()

      prob_pert                    = torch.softmax(out_data_pred_pertubarted, dim=1)
      pred_prob_pert_init_class    = prob_pert[0,class_pred_idx]

      mapper[batch_idx]['perturbated_predicitions_diff'].append(pred_prob -  pred_prob_pert_init_class)




if __name__ == "__main__":
  parser = argparse.ArgumentParser()


  parser.add_argument('--custom-trained-model',
                        required = True,
                        help='')
  parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

  parser.add_argument('--data-path')


  parser.add_argument('--variant', default = 'basic' , type=str, help="")
  parser.add_argument('--num-steps', default = 10 , type=int, help="")

  parser.add_argument('--class-index',
                       # default = "243",
                       type=int,
                        help='') #243 - dog , 282 - cat
  parser.add_argument('--method', type=str,
                        default='transformer_attribution',

                        help='')



  parser.add_argument('--fract', type=float,
                        default=0.1,
                        help='')
  args = parser.parse_args()
  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
  config.set_components_custom_lrp(args)




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

  mapper = {}

  basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

  root = os.path.join(args.data_path,  'val')
  dataset_val = datasets.ImageFolder(root, transform=basic_transform)


  np.random.seed(42)
  torch.manual_seed(42)
  total_size  = len(dataset_val)
  indices = list(range(total_size))
  subset_size = int(total_size *args.fract)
  random_indices = np.random.choice(indices, size=subset_size, replace=False)
  sampler = SubsetRandomSampler(random_indices)

  mapper = {}

  sample_loader = torch.utils.data.DataLoader(
      dataset_val, sampler=sampler,
      batch_size=1,
      shuffle=False,
      pin_memory=True,
      num_workers=1,
      drop_last = True
  )


  for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):

    epsilon_rule = args.epsilon_rule
    gamma_rule   = args.gamma_rule
    default_op   = args.default_op
    aggrage_info(batch_idx=batch_idx,original_image= data.squeeze(0),
                 mapper=mapper,args=args)




  final_res = {}

  count = 0
  count_pos = 0
  F_tot = 0
  F_tot_pos = 0
  for num in mapper:
   F = compute_SaCo(mapper,num,args.num_steps)
   count +=1
   F_tot+= F
   if F>0:
      F_tot_pos+=F
      count_pos+=  1
   final_res[str(num)] = F

  update_json(f"testing/res_{args.variant}_{args.method}_{args.num_steps}.json",
         {'F': f"{F_tot / count }"})
  update_json(f"testing/res_{args.variant}_{args.method}_{args.num_steps}.json",
         {'F_pos': f"{F_tot_pos / count_pos }"})
  update_json(f"testing/res_{args.variant}_{args.method}_{args.num_steps}.json",
         {'NUM_SAMPLES': f"{count}"})








# i *4 - 1

# 7  - black the patch from beginning | zero relevance from end - both models
# 11,10 , 12, 15 - perfect for zero down relevance from the end (for basic)
# 21, 20 - both zero down important from end, and black the most important ones from the start
# 25 - crucial for blacking out from the start (how the model behaves when there is no eye)
# 32 - crucial from zero relevnce from end
# 45 - zero relevance from end
# 57 - zero patch from start and from end - I want this to be our solution to overcondesniation - somewhere the model has to understand this is a bird
# 63 - black from start
# 65 - zero down relevance and check if somewhere else we understand its a shark - go over everything (attention, all heads, and lrp)
# 74 - also, check where does the model understands by structure that this is a mouse
# 131 - most extreme example - find out where is the bird
# 139 - also,zero down relevance

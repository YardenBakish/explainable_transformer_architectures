import os
import config
import argparse
import subprocess
#import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='evaluate perturbations')
    
    parser.add_argument('--mode', required=True, choices = ['perturbations', 'analyze'])
 

    parser.add_argument('--gen-latex', action='store_true')
    parser.add_argument('--num-steps', default = 10 , type=int, help="")
    parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

    parser.add_argument('--variant', default = 'basic',  type=str, help="")
  
    parser.add_argument('--fract', type=float,
                        default=0.1,
                        help='')
    parser.add_argument('--data-path', type=str,
                  
                        help='')
    parser.add_argument('--check-all', action='store_true')


    
    parser.add_argument('--num-workers', type=int,
                        default= 1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                  
                        help='')

 
    args = parser.parse_args()
    return args





MAPPER_HELPER = {
   'basic': r'\underline{DeiT-tiny}',
   'attn act relu': 'ReluAttetnion w/ cp',
   'act softplus':   'Softplus Act.',
   'act softplus norm rms': 'Softplus+RMSNorm',
   'norm rms': 'RMSNorm',
   'bias ablation': 'DeiT-tiny w/o Bias',
   'norm bias ablation': 'LayerNorm w/o Bias',
   'attn act sparsemax': 'Sparsemax',
   'variant layer scale': 'DeiT-tiny w/ LayerScale',
   'attn variant light': 'LightNet',
   'variant more ffn': '2XFFN',
   'variant more attn': '2XAttention',
   'variant simplified blocks': 'DeiT-tiny w/o normalization',
   'attn act sigmoid': 'SigmoidAttention',
   'attn act relu no cp': 'ReluAttention',
   'norm batch':           'RepBN (BatchNorm)',
   'custom_lrp': 'lrp',
   'transformer_attribution': 'transformer attribution',
   'variant weight normalization': 'WeightNormalization',
   'variant diff attn': 'DiffTransformer',
   'variant diff attn relu': 'DiffTransformer w/ Relu',
   'attn act relu pos' : 'ReluAttention w/ Softplus',
   'variant registers': 'DeiT-tiny w/ registers',
   'variant relu softmax': 'Relu m.w/ Softmax',
   'attn act relu normalized': 'Propotional Relu',
   'act relu': 'Relu Act.',
   'variant layer scale relu attn': 'ReluAttention w/ LayerScale',
   'variant more attn relu': '2xAttentionRelu',
   'variant patch embed relu': 'Large Emb w/ Relu',
   'variant patch embed': 'Large Emb',
   'variant drop high norms preAct': 'RandomDropPreAct',
   'variant drop high norms postAct': 'RandomDropPostAct',
   'variant drop high norms relu': 'RandomDropRelu',
   'custom_lrp_gamma_rule_default_op': r'(semi-$\gamma$-rule)',
   'custom_lrp_gamma_rule_full': r'($\gamma$-rule)',
   'lrp': r'($\alpha\beta$-rule)',


}


def gen_latex_table(global_top_mapper,args):
    latex_code = r'\begin{table}[h!]\centering' + '\n' + r'\begin{tabular}{c c c c c c}' + '\n' + '\hline' +'\n'
    latex_code += '& Pixel Accuracy & mAP & mIoU & mBGI & mFI' r'\\ ' +'\hline \n'

    variants_set = set()
    for elem in global_top_mapper:
      variant,epoch,pixAcc, mAP, mIoU, mBG_I,mFG_I, suffix = elem
      
      if variant not in variants_set:
         variants_set.add(variant)
      else:
         continue
      variant = variant.split("/")[-1]
      variant = variant.replace("_"," ")
      variant = f'{MAPPER_HELPER[variant]} {MAPPER_HELPER[suffix]}' 

      row = variant
      row += f' & {pixAcc:.3f}'
      row += f' & {mAP:.3f}'
      row += f' & {mIoU:.3f}'
      row += f' & {mBG_I:.3f}'
      row += f' & {mFG_I:.3f}'
      
      row += r'\\ ' f'\n'

      latex_code += row
    
    latex_code += "\\hline\n\\end{tabular}\n\\caption{Segmentation Results using}\n\\end{table}"

    print(latex_code)




d = {
       
        #BATCH 1

        'attn_act_relu_no_cp': [80,88],
        'basic':          [0],
        
        #BATCH 3
        'variant_diff_attn_relu': [155],
        'variant_layer_scale': [43],
        'attn_act_relu_pos': [145],

        'variant_more_attn_relu': [265],
        'variant_layer_scale_relu_attn': [100], #BATCH1
    }
   
   

'''
TEMPORARY! based on current accuarcy results
'''
def filter_epochs(args, epoch, variant):
   return epoch in args.epochs_to_perturbate[variant]



def run_perturbations_env(args):
   choices = args.epochs_to_perturbate.keys()
   methods = ["custom_lrp", 'custom_lrp_gamma_rule_default_op', 'custom_lrp_gamma_rule_full']
   for c in choices:
      for method in methods:
        args.method = method
        args.variant = c
        run_perturbation(args)
  



def get_sorted_checkpoints(directory):
    # List to hold the relative paths and their associated numeric values
    checkpoints = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the pattern 'checkpoint_*.pth'
            match = re.match(r'checkpoint_(\d+)\.pth', file)
            if match:
                # Extract the number from the filename
                number = int(match.group(1))
                # Get the relative path of the file
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                # Append tuple (number, relative_path)
                checkpoints.append((number, relative_path))

    # Sort the checkpoints by the number
    checkpoints.sort(key=lambda x: x[0])

    # Return just the sorted relative paths
    return [f'{directory}/{relative_path}'  for _, relative_path in checkpoints]


def run_perturbation(args):
    eval_pert_cmd        = "python run_pert_saco.py"
   
    
    eval_pert_cmd       +=  f' --method {args.method}'

  
    root_dir = f"{args.dirs['finetuned_models_dir']}{args.data_set}"
    
    variant          = f'{args.variant}'
    eval_pert_cmd += f' --variant {args.variant}'
    eval_pert_cmd += f' --fract {args.fract} '
    eval_pert_cmd += f' --num-steps {args.num_steps} '

   
  

    model_dir = f'{root_dir}/{variant}'

    checkpoints =  get_sorted_checkpoints(model_dir)


    for c in checkpoints:
     
       checkpoint_path  = c.split("/")[-1]
       epoch            = checkpoint_path.split(".")[0].split("_")[-1]
       if filter_epochs(args, int(epoch), variant ) == False:
          continue
       print(f"working on epoch {epoch}")
       eval_pert_cmd += f" --custom-trained-model {model_dir}/{checkpoint_path}" 
       print(f'executing: {eval_pert_cmd}')
       try:
          subprocess.run(eval_pert_cmd, check=True, shell=True)
          print(f"generated visualizations")
       except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
          exit(1)




def parse_seg_results(seg_results_path, method,variant, analyze_all_lrp = False):
    pass





def parse_subdir(subdir):
   exp_name = subdir.split("/")[-1]
   exp_name = exp_name.replace("_"," ")
   exp_name = exp_name if exp_name != "none" else "basic"
   return exp_name



def parse_filename(filename):
    all_methods = ['custom_lrp', 'custom_lrp_gamma_rule_full', 'custom_lrp_gamma_rule_default_op' ]

    # Strip the extension if it has one
    base_name = filename.rsplit('.', 1)[0]
    
    # Initialize a list to store the parsed parts
    parsed_parts = []

    # Start processing the filename from the end
    parts = base_name.split('_') # Reverse the parts to start from the end

    num_steps = parts[-1]
    parts = parts[:-1]
    method  = parts[-1]
    parts = parts[:-1]
    while method not in all_methods:
       method = parts[-1]+"_" +method
       parts = parts[:-1]
    variant = parts[-1]
    parts = parts[:-1]
    while len(parts) != 1:
       variant = parts[-1]+"_" +variant
       parts = parts[:-1]
    return variant, method, num_steps




def parse_test_results(filepath):
    
    with open(filepath, 'r') as f:
             data = json.load(f)
             F =  data.get(f'F',0)
             F_pos = data.get(f'F_pos',0)

             F = float(F)
             F_pos  = float(F_pos)
            
    return F, F_pos


def analyze(args):
   mapper = {}
   for filename in os.listdir("testing/"):
      if filename.startswith("res"):
        full_path = f"testing/{filename}"
        variant, method, num_steps = parse_filename(filename)
        #print(f"{variant} {method} {num_steps}")
        F, F_pos = parse_test_results(full_path)
        if num_steps not in mapper:
           mapper[num_steps] = []
        mapper[num_steps].append((variant, method, F,F_pos, num_steps))
    
   for k in mapper:
      mapper[k].sort(reverse=True, key=lambda x: x[2])
      seen_variant = set()
      for row in mapper[k]:
         variant, method, F, F_pos, num_steps = row
         if variant in seen_variant:
            continue
         else:
            seen_variant.add(variant)
         print(f"VARIANT: {variant} | method: {method} | F: {F} | F_pos: {F_pos} | STEPS: {num_steps}")
      print("\n\n")
    
        


   
if __name__ == "__main__":
    args                   = parse_args()
    config.get_config(args, skip_further_testing = True)
    args.epochs_to_perturbate = d
  
    if args.mode == "perturbations":
       if args.check_all:
          run_perturbations_env(args)
       else: 
         run_perturbation(args)
    else:
       analyze(args)
    
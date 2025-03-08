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
    
    parser.add_argument('--mode', required=True, choices = ['segmentations', 'analyze', 'analyze_comparison'])
 

    parser.add_argument('--gen-latex', action='store_true')
    parser.add_argument('--threshold-type', choices = ['mean', 'otsu', 'MoV'], required = True)
    parser.add_argument('--variant', default = 'basic',  type=str, help="")
  

    parser.add_argument('--check-all', action='store_true')
    parser.add_argument('--analyze-all-lrp', action='store_true')
    parser.add_argument('--analyze-all-full-lrp', action='store_true')


 
    parser.add_argument('--data-path', type=str,
                  
                        help='')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')

    parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)
    
    parser.add_argument('--num-workers', type=int,
                        default= 1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        help='')

 

    parser.add_argument('--imagenet-seg-path', type=str, default = "gtsegs_ijcv.mat",help='')
    args = parser.parse_args()
    return args


ALL_LRP_SUBDIRS = [#'lrp',
                  #'custom_lrp'
                   #'custom_lrp_gamma_rule_default_op', 
                   #'custom_lrp_gamma_rule_full' ,
                   #'custom_lrp_SEMANTIC_ONLY',
                   'custom_lrp_PE_ONLY',
                  #"custom_lrp_gamma_rule_full",
                  #"custom_lrp_gamma_rule_full_SEMANTIC_ONLY",
                  "custom_lrp_gamma_rule_full_PE_ONLY"
                   
                   
                   ]
ALL_FULL_LRP_SUBDIRS = [
   
   
   #'full_lrp_GammaLinear_POS_ENC_PE_ONLY_gammaConv'
   #'full_lrp_semiGammaLinear_alphaConv', 
                        #'full_lrp_GammaLinear_alphaConv',  
                        #'full_lrp_Linear_alphaConv', 
                        #'full_lrp_semiGammaLinear_gammaConv',
                        #'full_lrp_GammaLinear_gammaConv',  
                        #'full_lrp_Linear_gammaConv' ,
                        #'full_lrp_GammaLinear_POS_ENC_alphaConv'

                        #'full_lrp_GammaLinear_POS_ENC_gammaConv',
                        #'full_lrp_GammaLinear_POS_GRAD_ENC_gammaConv',
                        #  
                        #'full_lrp_semiGammaLinear_POS_ENC_alphaConv', ,  'full_lrp_Linear_POS_ENC_alphaConv',
                        #'full_lrp_semiGammaLinear_POS_ENC_gammaConv',   'full_lrp_Linear_POS_ENC_gammaConv',
                        #'full_lrp_GammaLinear_POS_GRAD_ENC_alphaConv',
                        #'full_lrp_semiGammaLinear_POS_GRAD_ENC_alphaConv',   'full_lrp_Linear_POS_GRAD_ENC_alphaConv',
                        #'full_lrp_semiGammaLinear_POS_GRAD_ENC_gammaConv',  'full_lrp_Linear_POS_GRAD_ENC_gammaConv',
                       
                       
                       ]


'''
 
'''


MAPPER_HELPER = {
   'basic': r'\underline{DeiT-tiny}',
   'attn act relu': 'ReluAttetnion w/ cp',
   'act softplus':   'Softplus Act.',
   'act softplus norm rms': 'Softplus+RMSNorm',
   'norm rms': 'RMSNorm',
   'bias ablation': 'DeiT-tiny w/o Bias(F)',
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
   'lrp': 'ours',
   'transformer_attribution': 'transformer attribution',
   'variant weight normalization': 'WeightNormalization',
   'variant diff attn': 'DiffTransformer',
   'variant diff attn relu': 'DiffTransformer w/ Relu',
   'attn act relu pos' : 'ReluAtt. w/Softplus w.o/bias(P)',
   'variant registers': 'DeiT-tiny w/ registers',
   'variant relu softmax': 'Relu m.w/ Softmax',
   'attn act relu normalized': 'Propotional Relu',
   'act relu': 'Relu Act.',
   'variant layer scale relu attn': 'ReluAttention w/ LayerScale',
   'variant more attn relu': '2xAttentionRelu',
   'variant patch embed relu': 'Large Emb w/ Relu w.o bias(T)',
   'variant patch embed': 'Large Emb w.o bias(T)',
   'variant drop high norms preAct': 'RandomDropPreAct',
   'variant drop high norms postAct': 'RandomDropPostAct',
   'variant drop high norms relu': 'RandomDropRelu',
   'custom_lrp_gamma_rule_default_op': r'(semi-$\gamma$-rule)',
   'custom_lrp_gamma_rule_full': r'($\gamma$-rule)',
   'attribution_w_detach': r'($\alpha\beta$-rule)',
   'custom_lrp': r'($\alpha\beta$-rule)',
   'imagenet_norm_no_crop': r'($\alpha\beta$-rule)',

   'base small':   'DeIT S',
   'basic medium': 'DeIT B',
   'medium relu attn': 'ReluAttention B w/o bias(P)',
   'small relu attn': 'ReluAttention S  w/o bias(P)',
   'variant complete patch embed relu': 'ReluAtt. w/Softplus w.o/bias(T)',
   'variant refined patch embed': 'Deit w/conv w.o/bias(T)',
   'variant refined patch embed relu': 'ReluAttSoft w/conv w.o/bias(T)',

   'variant full no bias relu': 'ReluAtt w.o/bias(T)',
   'variant no bias relu': 'ReluAtt w.o/bias(P)',
   'variant relu plus conv': 'ReluAtt w/conv w.o/bias(T)',
   'medium relu attn w bias' : 'ReluAtt B w/ bias',
   'small relu attn w bias':  'ReluAtt S w/ bias',
   'medium relu full no bias': 'ReluAtt B w.o/ bias(T)',
   'small relu full no bias':  'ReluAtt S w.o/ bias(T)',
   'variant relu relu':        'ReluAtt w/ Relu',
   'base small no bias':       'DeIT S w.o/ bias(T)',
   'basic medium no bias':     'DeIT B w.o/ bias(T)',

   'custom_lrp_SEMANTIC_ONLY' :'attnLRP',
   'custom_lrp_PE_ONLY': 'PA-LRP',
   "custom_lrp_gamma_rule_full": 'ours (gamma rule)',
   "custom_lrp_gamma_rule_full_SEMANTIC_ONLY" : 'attnLRP (gamma rule)',
   "custom_lrp_gamma_rule_full_PE_ONLY" : 'PA-LRP (gamma rule)',

   'full_lrp_semiGammaLinear_alphaConv': r'(semi-$\gamma,\alpha$)' , 
   'full_lrp_GammaLinear_alphaConv': r'($\gamma,\alpha$)',  
   'full_lrp_Linear_alphaConv' : r'($\alpha\beta,\alpha$)', 
   'full_lrp_semiGammaLinear_gammaConv': r'(semi-$\gamma,\gamma$)',
   'full_lrp_GammaLinear_gammaConv': r'($\gamma,\gamma$)',  
   'full_lrp_Linear_gammaConv': r'($\alpha\beta,\gamma$)',

   'gammaConv': r'($\alpha\beta,\gamma$)',
   'alphaConv': r'($\alpha\beta,\alpha$)',

   'full_lrp_semiGammaLinear_POS_ENC_alphaConv': r'(pe,semi-$\gamma,\alpha$)' , 
   'full_lrp_GammaLinear_POS_ENC_alphaConv': r'(pe,$\gamma,\alpha$)',  
   'full_lrp_Linear_POS_ENC_alphaConv' : r'(pe,$\alpha\beta,\alpha$)', 
   'full_lrp_semiGammaLinear_POS_ENC_gammaConv': r'(pe,semi-$\gamma,\gamma$)',
   'full_lrp_GammaLinear_POS_ENC_gammaConv': r'(pe,$\gamma,\gamma$)',  
   'full_lrp_Linear_POS_ENC_gammaConv': r'(pe,$\alpha\beta,\gamma$)',

   'full_lrp_GammaLinear_POS_ENC_PE_ONLY_gammaConv': 'PE ONLY',
   
   'full_lrp_semiGammaLinear_POS_GRAD_ENC_alphaConv': r'(pge,semi-$\gamma,\alpha$)' , 
   'full_lrp_GammaLinear_POS_GRAD_ENC_alphaConv': r'(pge,$\gamma,\alpha$)',  
   'full_lrp_Linear_POS_GRAD_ENC_alphaConv' : r'(pge,$\alpha\beta,\alpha$)', 
   'full_lrp_semiGammaLinear_POS_GRAD_ENC_gammaConv': r'(pge,semi-$\gamma,\gamma$)',
   'full_lrp_GammaLinear_POS_GRAD_ENC_gammaConv': r'(pge,$\gamma,\gamma$)',  
   'full_lrp_Linear_POS_GRAD_ENC_gammaConv': r'(pge,$\alpha\beta,\gamma$)',

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
      row += f' & {100*mAP:.3f}'
      row += f' & {100* mIoU:.3f}'
      row += f' & {mBG_I:.3f}'
      row += f' & {mFG_I:.3f}'
      
      row += r'\\ ' f'\n'

      latex_code += row
    
    latex_code += "\\hline\n\\end{tabular}\n\\caption{Segmentation Results using}\n\\end{table}"

    print(latex_code)





   
   

'''
TEMPORARY! based on current accuarcy results
'''
def filter_epochs(args, epoch, variant):
   return epoch in args.epochs_to_segmentation[variant]



def run_segmentations_env(args):
   choices = args.epochs_to_segmentation.keys()
   for c in choices:
      args.variant = c
      run_segmentation(args)
  



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


def run_segmentation(args):
    eval_seg_cmd        = "python evaluate_segmentation.py"
   
    
    eval_seg_cmd       +=  f' --method {args.method}'
    eval_seg_cmd       +=  f' --imagenet-seg-path {args.imagenet_seg_path}'
  

    root_dir = f"{args.dirs['finetuned_models_dir']}{args.data_set}"
    
    variant          = f'{args.variant}'
    eval_seg_cmd += f' --variant {args.variant}'
    eval_seg_cmd += f' --threshold-type {args.threshold_type} '
   
  

    model_dir = f'{root_dir}/{variant}'

    checkpoints =  get_sorted_checkpoints(model_dir)

    
    #CHANGEHERE
    suff = args.method 
    #suff = (args.method).split("_")[-1] if len(args.method) <26 else args.method

    for c in checkpoints:
     
       checkpoint_path  = c.split("/")[-1]
       epoch            = checkpoint_path.split(".")[0].split("_")[-1]
       if filter_epochs(args, int(epoch), variant ) == False:
          continue
       print(f"working on epoch {epoch}")
       seg_results_dir = 'seg_results2' if args.threshold_type == 'mean' else f'seg_results2_{args.threshold_type}'
       eval_seg_epoch_cmd = f"{eval_seg_cmd} --output-dir {model_dir}/{seg_results_dir}/res_{epoch}_{suff}"
       eval_seg_epoch_cmd += f" --custom-trained-model {model_dir}/{checkpoint_path}" 
       print(f'executing: {eval_seg_epoch_cmd}')
       try:
          subprocess.run(eval_seg_epoch_cmd, check=True, shell=True)
          print(f"generated visualizations")
       except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
          exit(1)




def parse_seg_results(seg_results_path, method,variant, analyze_all_lrp = False, analyze_all_full_lrp = False):
    suffixLst = [args.method]
    
    #suffixLst = [method.split("_")[-1] if len(args.method) <26 else args.method]
    if analyze_all_lrp:
       suffixLst = [elem for elem in ALL_LRP_SUBDIRS]
    if analyze_all_full_lrp:
       suffixLst = [elem for elem in ALL_FULL_LRP_SUBDIRS]
       for i in range(len(suffixLst)):
          suffixLst[i] =   suffixLst[i]
          #CHANGEHERE
          #suffixLst[i] = suffixLst[i].split("_")[-1] if len(suffixLst[i]) <26 else suffixLst[i]

    
   
    lst = []
   
    for res_dir in os.listdir(seg_results_path):
        for suffix in suffixLst:
         parsed_dir =  res_dir.split("_")
         if len(parsed_dir) < 3 or "_".join(parsed_dir[2:]) != suffix:
            continue
         
      
            
           
           
         epoch = int(res_dir.split('_')[1])



         seg_results_file = f'{seg_results_path}/{res_dir}/seg_results.json'
         with open(seg_results_file, 'r') as f:
             seg_data = json.load(f)
             mIoU =  seg_data.get(f'mIoU',0)
             pixAcc =  seg_data.get(f'Pixel Accuracy',0)
             mBG_I =  seg_data.get(f'mean_bg_intersection',0)
             mFG_I =  seg_data.get(f'mean_fg_intersection',0)
             mAP   =  seg_data.get(f'mAP',0)
             lst.append((variant ,epoch,float(pixAcc), float(mAP), float(mIoU), float(mBG_I),float(mFG_I), suffix ))
    return lst





def parse_subdir(subdir):
   exp_name = subdir.split("/")[-1]
   exp_name = exp_name.replace("_"," ")
   exp_name = exp_name if exp_name != "none" else "basic"
   return exp_name


def analyze(args):
   choices  =  args.epochs_to_segmentation.keys() 
   root_dir = f"{args.dirs['finetuned_models_dir']}{args.data_set}"

   global_lst = []
   seg_results_dir = 'seg_results2' if args.threshold_type == 'mean' else f'seg_results_{args.threshold_type}'


   for c in choices:
       subdir = f'{root_dir}/{c}/{seg_results_dir}'    
       global_lst += parse_seg_results(subdir, args.method,c, analyze_all_lrp=args.analyze_all_lrp, analyze_all_full_lrp = args.analyze_all_full_lrp)
    

   global_lst.sort(reverse=True, key=lambda x: x[4]) #mIOU

   print("oredered by mIOU")
   for elem in global_lst:
      variant,epoch,pixAcc, mAP, mIoU, mBG_I,mFG_I, suffix = elem
      print(f"variant: {variant} | e: {epoch} | mIoU: {mIoU} | pixAcc: {pixAcc} | mAP: {mAP}")
   
   
   if args.gen_latex:
      gen_latex_table(global_lst,args)



def analyze_comparison(args):
    filepath = "testing/seg_results_compare.json"
    # Read the JSON file
    with open(filepath, 'r') as file:
       data = json.load(file)
    x_values = [0, 0.2, 0.4, 0.6, 0.8]
    d_res  = {}
    # Loop through the first-level keys (groups)
    for variant, epochs_dict in data.items():
     for epoch, experiments in epochs_dict.items():
        for experiment_name, values in experiments.items(): 
         if experiment_name not in d_res:
             d_res[experiment_name]  ={}
         d_res[experiment_name][f'{variant}_{epoch}'] = values
                # Create a plot for each experiment
    
     for experiment, results in d_res.items():
         plt.figure(figsize=(10, 6))
         plt.title(f"Experiment: {experiment}")
         for test, values  in results.items():
            plt.plot(x_values,values, label=test)
            plt.xlabel('eps')
            plt.ylabel('score')
            plt.legend()
         plt.savefig(f'testing/{experiment}.png')
         plt.close()  

        


   
if __name__ == "__main__":
    args                   = parse_args()
    config.get_config(args, skip_further_testing = True, get_epochs_to_segmentation = True)
    
    if args.mode == 'analyze_comparison':
       analyze_comparison(args)
    
      
    elif args.mode == "segmentations":
       if args.check_all:
          run_segmentations_env(args)
       else: 
         run_segmentation(args)
    else:
       analyze(args)
    
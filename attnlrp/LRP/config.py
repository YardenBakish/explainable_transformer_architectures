from functools import partial



DEFAULT_PATHS = {
    'original_models_dir'    : 'original_models/', 
    'finetuned_models_dir'   : 'finetuned_models/', 
    'results_dir_root'       : 'finetuned_models/', 

}





#chosen randmoly
EPOCHS_TO_PERTURBATE = {

}





MODEL_VARIANTS = {
    'baseline': {'components_with_grad': {'all': True, 'components': ["score.weight"]}},
    'baseline2': {'components_with_grad': {'all': False, 'components': ["score.weight"]}},

    'attn_act_relu': {'components_with_grad': {'all': True, 'components': []}},
    'attn_act_relu2': {'components_with_grad': {'all': True, 'components': []}}, #
    'attn_act_relu3': {'components_with_grad': {'all': True, 'components': []}},
    'attn_act_relu4': {'components_with_grad': {'all': True, 'components': []}},


    'attn_act_sigmoid': {'components_with_grad': {'all': True, 'components': []}},
    'attn_act_sigmoid2': {'components_with_grad': {'all': True, 'components': []}},

    'baseline_full': {'components_with_grad': {'all': False, 'components': ["score.weight"], 'partial_unfreeze': 21}},


}


   
def get_config(args, pert = False):
    if args.variant == 'baseline2':
        args.quant = True

    args.original_models       = f"{DEFAULT_PATHS['original_models_dir']}{args.model_size}/vanilla" 
    args.pretrained_model_path = f"{DEFAULT_PATHS['results_dir_root']}{args.dataset}/{args.model_size}/{args.variant}" 
    if args.variant not in MODEL_VARIANTS:
        print("must choose existing variant")
        exit(1)
    args.model_components = MODEL_VARIANTS[args.variant]

    args.attn_layer = args.variant



    if 'attn_act_relu' in args.attn_layer:
        args.attn_layer = 'attn_act_relu'
    if 'attn_act_sigmoid' in args.attn_layer:
        args.attn_layer = 'attn_act_sigmoid'

    if args.sequence_length == None:
        args.sequence_length = 4096 if args.model_size == 'llama_2_7b' else 2048



    if pert:
        return
    if args.finetune:
        if args.finetune not in MODEL_VARIANTS:
            print("must choose existing variant to finetune")
            exit(1)
        else: 
            args.finetuned_attn_layer = args.finetune
            args.finetuned_model_path = f"{DEFAULT_PATHS['results_dir_root']}{args.dataset}/{args.model_size}/{args.finetune}" 


    if args.finetune and args.resume:
        print("can either only resume or only finetune (finetune is the start)")
        exit(1)



    

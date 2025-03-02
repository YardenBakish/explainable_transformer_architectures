

#finetune deit from scratch
python main.py --auto-save --results-dir finetuned_models  --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1

#continue finetuning basic
python main.py --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#evaluate basic
python main.py --eval --resume finetuned_models/basic/basic.pth --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1


#train ablated component fron finetuned
python main.py --is-ablation --ablated-component bias --auto-save --results-dir finetuned_models   --finetune finetuned_models/basic/basic.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1




#evaluate perturbations
python evaluate_perturbations.py --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1 --custom-trained-model finetuned_models/none/best_checkpoint.pth --num-workers 1 --both

# full_lrp


#evaluate ablation
python main.py --ablated-component bias --eval --auto-resume --results-dir finetuned_models --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1

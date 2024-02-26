# pfcrt_project

[Colab link](https://colab.research.google.com/drive/1dg0OAJAQt-rwmWPKFcv2CyK5Si8JIfyu?usp=sharing)

# Run to train (PyTorch > 2.0.0) [as of Feb 26th 2024]
```bash
# [PRETRAINING]
# For pretraining and saving the best checkpoint as best_pretrained.ckpt
git pull && python -m main --ngpus auto --accelerator gpu --strategy auto -b 512 --loss_weights 0 0 0

# [FINETUNING]
# For finetuning and continuing the best checkpoint saved as best_pretrained.ckpt
# Assume we save best_finetune.ckpt with SECOND target loss (i.e., --loss_weights 0 1 0) inside output_finetune_l1 directory
git pull && python -m main --ngpus auto --accelerator gpu --strategy auto -b 512 --finetune --loss_weights 0 1 0 --load_model_directory output_finetune_l1 --load_model_checkpoint best_pretrained.ckpt

# [INFERENCE]
export WANDB_DIR=/Scr/hyunpark/DL_Sequence_Collab/pfcrt_project #For W&B
export WANDB_CACHE_DIR=/Scr/hyunpark/DL_Sequence_Collab/pfcrt_project #For W&B
export TRANSFORMERS_CACHE=/Scr/hyunpark/DL_Sequence_Collab/pfcrt_project #For BertModel
git pull && python -m utils --ngpus 1 --accelerator gpu --strategy auto -b 512 --finetune --loss_weights 0 1 0 --load_model_directory output_finetune_l1 --load_model_checkpoint best_finetune.ckpt
```

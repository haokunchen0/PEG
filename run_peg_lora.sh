#!/bin/bash

# CIFAR10上使用PEG-LoRA微调DeiT-Tiny (单卡训练，使用第2张卡)
echo "Starting PEG-LoRA finetuning on CIFAR10..."
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12346 main.py \
  --cfg configs/deit_tiny_cifar10_lora.yaml \
  --finetune ./output/deit_tiny_lr=5e-4_warm=5_attn2to3_ffn6to12_initialize_100epochs/ckpt_best.pth

# CIFAR100上使用PEG-LoRA微调DeiT-Tiny (单卡训练，使用第3张卡)
echo "Starting PEG-LoRA finetuning on CIFAR100..."
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12347 main.py \
  --cfg configs/deit_tiny_cifar100_lora.yaml \
  --finetune ./output/deit_tiny_lr=5e-4_warm=5_attn2to3_ffn6to12_initialize_100epochs/ckpt_best.pth

echo "PEG-LoRA finetuning completed!" 
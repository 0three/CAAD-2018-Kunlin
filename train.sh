#!/bin/bash
# Following arguments has to be specified for training:
# - MAX_NUMBER_OF_TRAINING_STEPS - maximum number of training steps,
#     omit this flag or set it to -1 to have unlimited number of training steps.
# - MODEL_NAME - name of the model, now only "resnet_v2_50" is supported.
# - MOVING_AVG_DECAY - decay rate for exponential moving average of the
#     trainable variables. Training with exponential moving average usually
#     leads to better accuracy. Default of 0.9999. -1 disable exponential moving
#     average. Default works well, so typically you set it only if you want
#     to disable this feature.
# - HYPERPARAMETERS - string with hyperparameters,
#     see model_lib.py for full list of hyperparameters.
# - DATASET - dataset, either "imagenet" or "tiny_imagenet".
# - IMAGE_SIZE - size of the image (single number).
# - OUTPUT_DIRECTORY - directory where to write results.
# - IMAGENET_DIR - directory with ImageNet dataset in TFRecord format.
# - TINY_IMAGENET_DIR - directory with Tiny ImageNet dataset in TFRecord format.
#
# Note that only one of IMAGENET_DIR or TINY_IMAGENET_DIR has to be provided
# depending on which dataset you use.
#
# Usage:
# ./run.sh JOBNAME PARTITION GPU_NUM
#
# Example:
# ./run.sh D3 AD1 32

set -x
set -e

JOBNAME=$1
PARTITION=$2
GPU_NUM=$3

TIMESTAMP="`date +%Y-%m-%d-%H-%M-%S`"

MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
    --job-name=${JOBNAME} \
    --partition=${PARTITION} \
    --mpi=pmi2 --gres=gpu:1 -n${GPU_NUM} --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    python /mnt/lustre/liukunlin/caad/alp/train.py \
    --model_name="resnet_v2_50" \
    --hparams="train_adv_method=pgdll_32_10_30,train_lp_weight=0.45" \
    --dataset="imagenet" \
    --dataset_image_size=299 \
    --output_dir="/mnt/lustre/liukunlin/caad/alp/pgd-image-299-newtry_normal-lvbo-" \
    --imagenet_data_dir="/mnt/lustre/liukunlin/caad/alp/imagenet-tfrecord" \
    --finetune_exclude_pretrained_scopes="resnet_v2_50/logits" \
    --finetune_trainable_scopes="resnet_v2_50/logits,resnet_v2_50/postnorm" \
    --finetune_checkpoint_path="/mnt/lustre/liukunlin/caad/alp/pretrained" 2>&1 | tee log_${TIMESTAMP} &

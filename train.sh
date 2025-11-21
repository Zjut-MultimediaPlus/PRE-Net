# Training Stage 1 for classification model
python train_swath_distilling.py \
  --eppoch 200 \
  --batch_size 32 \
  --mask_ratio 0.25 \
  --load_teacher PRE-demo/checkpoints/cls_teacher.pth \
  --classes 2 \

# Training Stage 1 for regression model
python train_swath_distilling.py \
  --eppoch 200 \
  --batch_size 32 \
  --mask_ratio 0.25 \
  --load_teacher PRE-demo/checkpoints/reg_teacher.pth \
  --classes 1 \

# Training Stage 2 for classification model
torchrun --nproc_per_node=4 train_full_disc_adaptation.py \
  --eppoch 100 \
  --mask-strategy self-masktune \
  --mask_ratio 0.25 \
  --fe-path cls_checkpoint_best.pth \
  --classes 2 \


# Training Stage 2 for regression model
torchrun --nproc_per_node=4 train_full_disc_adaptation.py \
  --eppoch 100 \
  --mask-strategy self-masktune \
  --mask_ratio 0.25 \
  --fe-path reg_checkpoint_best.pth \
  --classes 1 \

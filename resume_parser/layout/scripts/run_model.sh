echo "#0: efficientnet_b0_imagenet_ub-1_dicretized_adam_lr0.005_wd0.0001_ep20_bs64"
python3 train.py \
    --dataset_split     "/home/mwerner/Git/resume_parsing/resources/layout/split_{}.conf" \
    --num_splits        5 \
    --model             efficientnet_b0 \
    --pretrained        imagenet \
    --unfreeze_blocks   -1 \
    --discretized_image True \
    --optimizer         adam \
    --lr                0.005 \
    --weight_decay      0.0001 \
    --momentum          -1 \
    --num_train_epochs  20 \
    --train_batch_size  64 \
    --test_batch_size   256 \
    --save_mode         True

echo "#2: efficientnet_b0_None_ub-1_dicretized_adam_lr0.001_wd0.0001_ep50_bs64"
python3 train.py \
    --dataset_split     "/home/mwerner/Git/resume_parsing/resources/layout/split_{}.conf" \
    --num_splits        5 \
    --model             efficientnet_b0 \
    --pretrained        None \
    --unfreeze_blocks   -1 \
    --discretized_image True \
    --optimizer         adam \
    --lr                0.001 \
    --weight_decay      0.0001 \
    --momentum          -1 \
    --num_train_epochs  50 \
    --train_batch_size  64 \
    --test_batch_size   256 \
    --save_mode         True

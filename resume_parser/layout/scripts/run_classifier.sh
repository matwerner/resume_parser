echo "#1: efficientnet_b0_imagenet_discretized"
python3 train_classifier.py \
    --dataset_split     "/home/mwerner/Git/resume_parsing/resources/layout/split_{}.conf" \
    --num_splits        5 \
    --model             efficientnet_b0 \
    --pretrained        imagenet \
    --discretized_image True \
    --batch_size        256 \

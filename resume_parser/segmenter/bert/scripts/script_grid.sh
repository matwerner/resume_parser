BERT_MODEL='neuralmind/bert-base-portuguese-cased'
# BERT_MODEL='bert-base-multilingual-cased'
LABEL_FILE='/home/mwerner/Git/paper/resume_parser/resources/segmenter/labels.txt'
DATASET_SPLIT='/home/mwerner/Git/paper/resume_parser/resources/segmenter/split_{0}.conf'
NUM_SPLITS=5

FIXED_DATA_PARAMETERS="--labels_file ${LABEL_FILE} --dataset_split ${DATASET_SPLIT} --num_splits ${NUM_SPLITS}"
FIXED_MODEL_PARAMETERS="--bert_model ${BERT_MODEL} --train_batch_size 8 --num_train_epochs 10 --num_consecutive_worse_threshold 3 --save_model"

for lr in "0.000020" "0.000010" "0.000005"; do
	python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --learning_rate ${lr}
	python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --learning_rate ${lr} --extra_classifier_features
done

for lr in "0.000020" "0.000010" "0.000005"; do
	python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --learning_rate ${lr} --use_crf
	python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --learning_rate ${lr} --use_crf --extra_classifier_features
done


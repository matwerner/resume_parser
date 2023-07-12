BERT_MODEL='neuralmind/bert-base-portuguese-cased'
# BERT_MODEL='bert-base-multilingual-cased'
LABEL_FILE='/home/mwerner/Git/paper/resume_parser/resources/segmenter/labels.txt'
DATASET_SPLIT='/home/mwerner/Git/paper/resume_parser/resources/segmenter/split_{0}.conf'
NUM_SPLITS=5

FIXED_DATA_PARAMETERS="--labels_file ${LABEL_FILE} --dataset_split ${DATASET_SPLIT} --num_splits ${NUM_SPLITS} --fp16"
FIXED_MODEL_PARAMETERS="--bert_model ${BERT_MODEL} --train_batch_size 8 --num_train_epochs 10 --pooler concat --classifier_lr 0.001 --num_consecutive_worse_threshold 3 --save_model"

python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --fit_classifier_only

python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --fit_classifier_only --use_crf

python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --fit_classifier_only --extra_classifier_features

python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --fit_classifier_only --extra_classifier_features --use_crf

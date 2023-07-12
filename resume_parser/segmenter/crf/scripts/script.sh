LABEL_FILE='/home/mwerner/Git/paper/resume_parser/resources/segmenter/labels.txt'
SECTION_FILE='/home/mwerner/Git/paper/resume_parser/resources/segmenter/section_names_map.json'
DATASET_SPLIT='/home/mwerner/Git/paper/resume_parser/resources/segmenter/split_{0}.conf'
NUM_SPLITS=5

FIXED_DATA_PARAMETERS="--labels_file ${LABEL_FILE} --section_names_file ${SECTION_FILE} --dataset_split ${DATASET_SPLIT} --num_splits ${NUM_SPLITS}"
FIXED_MODEL_PARAMETERS="--num_jobs -3 --num_iter 50 --save_model"

# Nothing
python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --disable_spatial --disable_visual --disable_text

# Text
python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --disable_spatial --disable_visual

# Visual
# python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --disable_spatial --disable_text

# Spatial
# python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --disable_visual --disable_text

# Visual + Spatial
# python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --disable_text

# Visual + Text
python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --disable_spatial

# Spatial + Text
# python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} --disable_visual

# Spatial + Text + Visual
python3 train.py ${FIXED_DATA_PARAMETERS} ${FIXED_MODEL_PARAMETERS} 

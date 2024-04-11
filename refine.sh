export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LAYOUT_DIR=''
export LAYOUT_NAME=''
export LOG_DIR=''
export CHECKPOINT_DIR=''

export MIDAS_PATH=''
export ONEFORMER_PATH=''

export SD_MODEL_PATH=''
export CONTROLNET_MODEL_PATH=''
export CLIP_MODEL_PATH=''
export DEPTH_MODEL_PATH=''
export SEMANTIC_MODEL_PATH=''

export COARSE_MODEL_PATH=''

CUDA_VISIBLE_DEVICES=0 \
python main.py \
--log_dir ${LOG_DIR}/${LAYOUT_NAME}-${TIME} \
--checkpoint_dir ${CHECKPOINT_DIR}/${LAYOUT_NAME}-${TIME} \
--sd_model_path ${SD_MODEL_PATH} \
--controlnet_model_path ${CONTROLNET_MODEL_PATH} \
--clip_model_path ${CLIP_MODEL_PATH} \
--depth_model_path ${DEPTH_MODEL_PATH} \
--semantic_model_path ${SEMANTIC_MODEL_PATH} \
--layout_path ${LAYOUT_DIR}/${LAYOUT_NAME}.npy \
--finetune True \
--resume ${COARSE_MODEL_PATH} 
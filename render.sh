export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LAYOUT_DIR=''
export LAYOUT_NAME=''
export LOG_DIR=''
export CHECKPOINT_DIR=''

export PRETRAINED_MODEL=''

CUDA_VISIBLE_DEVICES=0 \
python main.py \
--log_dir ${LOG_DIR}/${LAYOUT_NAME}-${TIME} \
--checkpoint_dir ${CHECKPOINT_DIR}/${LAYOUT_NAME}-${TIME} \
--layout_path ${LAYOUT_DIR}/${LAYOUT_NAME}.npy \
--resume ${PRETRAINED_MODEL} \
--render_video True
MESH='disk'
OUT_NAME='test_dream.obj'
CFG_PATH='../config/style_transfer_train.yml'
MODEL_PATH='../results/model/exp_03_17_00_51_59_10classes'

python deepdream_loop.py \
  --which_starting_mesh ${MESH} \
  --output_filename ${OUT_NAME} \
  --config_path ${CFG_PATH} \
  --model_path ${MODEL_PATH}
  
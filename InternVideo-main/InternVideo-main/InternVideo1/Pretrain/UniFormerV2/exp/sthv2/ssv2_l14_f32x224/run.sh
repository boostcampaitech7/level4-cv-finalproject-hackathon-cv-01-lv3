NUM_SHARDS=4
NUM_GPUS=8
BATCH_SIZE=32
BASE_LR=2e-5
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net_multi_node.py \
  --init_method tcp://localhost:10125 \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR ./data_list/sthv2 \
  DATA.PATH_PREFIX you_data_path/sthv2 \
  DATA.LABEL_PATH_TEMPLATE "somesomev2_rgb_{}_split.txt" \
  DATA.IMAGE_TEMPLATE "img_{:05d}.jpg" \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 50 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 22 \
  SOLVER.BASE_LR $BASE_LR \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 5. \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST True \
  DATA.MC True \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path

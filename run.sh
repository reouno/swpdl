#!/bin/sh


docker run \
    -v /home/leo/src/tensorflow-gpu:/workspace \
    -v /home/leo/src/dataset:/dataset \
    -it \
    --runtime=nvidia \
    --rm \
    tf:2.0.0a \
    python /workspace/swpdl/train.py \
        --conf /workspace/swpdl/stanford_dogs_config.yaml \
        --network mobilenet_v2_scratch \
        --save_dir /workspace/swpdl_logs/mobilenet_v2_scratch_dogs_190530 \
        --gpus 0 1

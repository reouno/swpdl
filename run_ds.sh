#!/bin/bash


docker run \
    -v /home/leo/src/tensorflow-gpu:/workspace \
    -v /home/leo/src/dataset:/dataset \
    -it \
    --runtime=nvidia \
    --rm \
    tf:2.0.0a \
    python /workspace/swpdl/distributed_training_sample.py

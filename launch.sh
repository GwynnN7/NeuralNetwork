#!/bin/bash

RELEASE_DIR="build/Release"

mkdir -p $RELEASE_DIR
cmake -B $RELEASE_DIR -DCMAKE_BUILD_TYPE=Release
cmake --build $RELEASE_DIR -j 6

# ./$RELEASE_DIR/NeuralNet xor \
#     --epochs 5000 --eta 0.5 --batch_size 4 \
#     --output relu \
#     --hidden sigmoid \
#     --network 2

./$RELEASE_DIR/NeuralNet mnist \
    --epochs 1000 --eta 0.3 --batch_size 1000 --lambda 1e-6 \
    --dataset_ratio 0.5 \
    --output sigmoid \
    --hidden sigmoid \
    --network 128 64

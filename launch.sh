#!/bin/bash

BUILD_DIR="build"

cmake -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release
cmake --build $BUILD_DIR -j 6

# ./$BUILD_DIR/NeuralNet xor \
#     --epochs 5000 --eta 0.5 --batch 4 \
#     --output relu \
#     --hidden sigmoid \
#     --network 2

./$BUILD_DIR/NeuralNet mnist \
    --epochs 1000 --eta 0.3 --batch 1000 --lambda 1e-6 \
    --dataset_ratio 0.5 \
    --output sigmoid \
    --hidden sigmoid \
    --network 128 64

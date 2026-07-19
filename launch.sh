#!/bin/bash

BUILD_DIR="build"

cmake --build $BUILD_DIR/

./$BUILD_DIR/NeuralNet xor_hot \
    --epochs 5000 --eta 0.3 --batch 4 \
    --output linear \
    --hidden relu \
    --network 3

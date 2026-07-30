#!/bin/bash

RELEASE_DIR="build/Release"

mkdir -p $RELEASE_DIR
cmake -B $RELEASE_DIR -DCMAKE_BUILD_TYPE=Release
cmake --build $RELEASE_DIR -j 6

./$RELEASE_DIR/NeuralNet xor_hot \
    --name xor_hot --train \
    --params dataset/xor/grid.csv \
    --epochs 2000

./$RELEASE_DIR/NeuralNet mnist \
    --name mnist \
    --epochs 200 \
    --dump --train \
    --params dataset/grid.csv \
    --dataset_ratio 0.4 --train_ratio 0.85 \
    --inner-k 2 --outer-k 2

./$RELEASE_DIR/NeuralNet mnist \
    --name mnist \
    --dataset_ratio 0.4 --train_ratio 0.85
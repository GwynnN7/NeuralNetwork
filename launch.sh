#!/bin/bash

RELEASE_DIR="build/Release"

mkdir -p $RELEASE_DIR
cmake -B $RELEASE_DIR -DCMAKE_BUILD_TYPE=Release
cmake --build $RELEASE_DIR -j 6

./$RELEASE_DIR/NeuralNet xor_hot \
    --name xor_hot --train \
    --params dataset/xor/grid.csv \
    --epochs 2000

booster ./$RELEASE_DIR/NeuralNet mnist \
    --name mnistl \
    --epochs 500 \
    --train \
    --params dataset/mnist/grid.csv \
    --dataset_ratio 1.0 --train_ratio 0.85 \
    --inner-k 1 --outer-k 1

./$RELEASE_DIR/NeuralNet mnist \
    --name mnist \
    --dataset_ratio 0.4 --train_ratio 0.85
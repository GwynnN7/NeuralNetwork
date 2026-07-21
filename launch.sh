#!/bin/bash

RELEASE_DIR="build/Release"

mkdir -p $RELEASE_DIR
cmake -B $RELEASE_DIR -DCMAKE_BUILD_TYPE=Release
cmake --build $RELEASE_DIR -j 6

./$RELEASE_DIR/NeuralNet xor_hot \
    --epochs 2000 --eta 0.5 --batch_size 1 \
    --init lecun \
    --output sigmoid \
    --hidden sigmoid \
    --network 3 \
    --dump dataset/dumps/xor.bin

./$RELEASE_DIR/NeuralNet xor_hot \
    --load dataset/dumps/xor.bin

./$RELEASE_DIR/NeuralNet mnist \
    --epochs 500 --eta 0.3 --batch_size 100 --lambda 1e-6 \
    --dataset_ratio 0.3 \
    --init he \
    --output softmax \
    --hidden relu \
    --network 128 64 \
    --dump dataset/dumps/mnist.bin

./$RELEASE_DIR/NeuralNet mnist \
    --load dataset/dumps/mnist.bin
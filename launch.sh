#!/bin/bash

RELEASE_DIR="build/Release"

mkdir -p $RELEASE_DIR
cmake -B $RELEASE_DIR -DCMAKE_BUILD_TYPE=Release
cmake --build $RELEASE_DIR -j 6

./$RELEASE_DIR/NeuralNet xor_hot \
    --name xor_hot --dump \
    --epochs 2000 --eta 0.5 --batch_size 1 \
    --init lecun \
    --output sigmoid \
    --hidden sigmoid \
    --network 3

./$RELEASE_DIR/NeuralNet xor_hot \
    --name xor_hot --load

./$RELEASE_DIR/NeuralNet mnist \
    --name mnist --dump \
    --epochs 500 --eta 0.3 --batch_size 100 --lambda 1e-6 \
    --dataset_ratio 0.3 --train_ratio 0.85 \
    --init he \
    --output softmax \
    --hidden relu \
    --network 128 64

./$RELEASE_DIR/NeuralNet mnist \
    --name mnist --load \
    --dataset_ratio 0.3

./$RELEASE_DIR/NeuralNet mnist \
    --name mnist_kfold \
    --dataset_ratio 0.4 --train_ratio 0.85 --kfold 2 \
    --epochs 500 --eta 0.4 --batch_size 400 --lambda 1e-6 \
    --init he \
    --output softmax \
    --hidden relu \
    --network 128 64
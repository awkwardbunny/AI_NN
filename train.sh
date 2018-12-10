#!/bin/bash
read -p "Enter the filename that contains the untrained ANN: " untrained
read -p "Enter the dataset file to train with: " data
read -p "Enter the trained output ANN filename: " trained
read -p "Enter learning rate: " alpha
read -p "Enter the number of iterations: " epochs

echo "Running NN with the command:"
echo "./nn.py train -inf $untrained -set $data -out $trained -alpha $alpha -epochs $epochs"
echo ""
./nn.py train -inf $untrained -set $data -out $trained -alpha $alpha -epochs $epochs

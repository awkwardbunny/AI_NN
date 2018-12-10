#!/bin/bash
read -p "Enter the filename that contains the trained ANN: " trained
read -p "Enter the dataset file to test with: " data
read -p "Enter the performance output filename: " out

echo "Running NN with the command:"
echo "./nn.py test -inf $trained -set $data -out $out"
echo ""
./nn.py test -inf $trained -set $data -out $out

#!/usr/bin/env python3

import sys
import argparse

class NeuralNet:
    def load(self, input_fn):
        print('Loading NN from {}'.format(input_fn))

    def save(self, output_fn):
        print('Saving NN to {}'.format(output_fn))

    def test(self, data_input_fn, output_fn):
        print('Testing NN from {} dataset and saving output to {}'.format(data_input_fn, output_fn))

    def train(self, data_input_fn, rate, epochs):
        print('Training NN from {} dataset at {} rate for {} iterations'.format(data_input_fn, rate, epochs))

def train(args):
    #print('Training NN using:\n {} as input\n {} as dataset\n {} as output\n {} learning rate\n {} iterations'
    #        .format(args.inf, args.set, args.out, args.alpha, args.epochs))

    nn = NeuralNet()
    nn.load(args.inf)
    nn.train(args.set, args.alpha, args.epochs)
    nn.save(args.out)

def test(args):
    #print('Testing NN using:\n {} as input\n {} as dataset\n {} as output'
    #        .format(args.inf, args.set, args.out))

    nn = NeuralNet()
    nn.load(args.inf)
    nn.test(args.set, args.out)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural Net Thingy', prog=sys.argv[0])
    subparsers = parser.add_subparsers(help="Select train or test", dest='action')
    subparsers.required = True

    parser_train = subparsers.add_parser("train", help="Train a neural net")
    parser_train.add_argument('-inf', help='Untrained input NN file', required = True)
    parser_train.add_argument('-set', help='Training set file', required = True)
    parser_train.add_argument('-out', help='Trained output NN file', required = True)
    parser_train.add_argument('-alpha', help='Learning rate', default = 0)
    parser_train.add_argument('-epochs', help='Number of iterations', default = 0)

    parser_test = subparsers.add_parser("test", help="Test a neural net")
    parser_test.add_argument('-inf', help='Trained input NN file', required = True)
    parser_test.add_argument('-set', help='Testing set file', required = True)
    parser_test.add_argument('-out', help='Output file', required = True)

    args = parser.parse_args()
    
    if(args.action == 'train'):
        train(args)
    else:
        test(args)

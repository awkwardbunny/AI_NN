#!/usr/bin/env python3

import sys
import argparse
#from collections import namedtuple
from recordtype import recordtype

Link = recordtype('Link', "weight f t")
Node = recordtype('Node', "sum output")

class NeuralNet:

    nums = None
    Nodes = None
    Links = None

    def load(self, input_fn):
        print('Loading NN from "{}"'.format(input_fn))
        with open(input_fn, 'r') as inf:
            self.nums = [int(i) for i in inf.readline().split()]
            self.Nodes = [None]*(sum(self.nums)+1) # +1 for constant 1 node of bias
            self.Links = [None]*((self.nums[0] + self.nums[2]) * self.nums[1] + self.nums[1] + 1) # nums[2]+1 for bias links

            # Add bias node (constant output 1 at the end of the list)
            self.Nodes[-1] = Node(-1, 1)

            # Populate input nodes
            for i in range(self.nums[0]):
                self.Nodes[i] = Node(-1, 0)

            # Populate hidden nodes and input/hidden links
            for i in range(self.nums[1]):
                line = [float(j) for j in inf.readline().split()]
                bias = line[0]
                weights = line[1:]

                self.Nodes[i+self.nums[0]] = Node(0, 0) # add current node
                self.Links[i*(self.nums[0]+1)] = Link(bias, -1, i+self.nums[0]) # bias node link
                for j in range(self.nums[0]): # links
                    self.Links[j+i*(self.nums[0]+1)+1] = Link(weights[j], j, i+self.nums[0])

            for i in range(self.nums[2]):
                line = [float(j) for j in inf.readline().split()]
                bias = line[0]
                weights = line[1:]

                self.Nodes[i+self.nums[0]+self.nums[1]] = Node(0, 0) # add current node
                self.Links[(i+1)*((self.nums[0]+1)*self.nums[1])] = Link(bias, -1, i+self.nums[0]+self.nums[1]) # bias node link
                for j in range(len(weights)): # links
                    self.Links[j+(i+1)*(self.nums[1]*(self.nums[0]+1))+1] = Link(weights[j], j+self.nums[0], i+self.nums[0]+self.nums[1])

    def save(self, output_fn):
        print('Saving NN to "{}"'.format(output_fn))
        with open(output_fn, 'w') as outf:
            outf.write(' '.join(str(x) for x in self.nums))
            outf.write('\n')

            for x in range(self.nums[1]):
                outf.write(' '.join(['{:.3f}'.format(self.Links[i].weight) for i in range((self.nums[0]+1)*x,(self.nums[0]+1)*(x+1))]))
                outf.write('\n')
    
            for x in range(self.nums[2]):
                outf.write(' '.join(['{:.3f}'.format(self.Links[i].weight) for i in range(
                    (self.nums[0]*self.nums[1]+self.nums[1])*(x+1),
                    (self.nums[0]*self.nums[1]+self.nums[1])*(x+1)+self.nums[1]+1
                    )]))
                outf.write('\n')
    
    def test(self, data_input_fn, output_fn):
        print('Testing NN from "{}" dataset and saving output to "{}"'.format(data_input_fn, output_fn))

    def train(self, data_input_fn, rate, epochs):
        print('Training NN from "{}" dataset at {} rate for {} iterations'.format(data_input_fn, rate, epochs))

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
    parser_train.add_argument('-alpha', help='Learning rate', default = 0.1)
    parser_train.add_argument('-epochs', help='Number of iterations', default = 100)

    parser_test = subparsers.add_parser("test", help="Test a neural net")
    parser_test.add_argument('-inf', help='Trained input NN file', required = True)
    parser_test.add_argument('-set', help='Testing set file', required = True)
    parser_test.add_argument('-out', help='Output file', required = True)

    args = parser.parse_args()
    
    if(args.action == 'train'):
        train(args)
    else:
        test(args)

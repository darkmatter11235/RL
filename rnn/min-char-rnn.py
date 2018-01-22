"""
Minimum character level RNN model
"""

import numpy as np

data = open('input.txt', 'r').read()
chars = list(set(data))
#print set(data)
#print "This is the chars"
#print chars
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' %(data_size, vocab_size)

char_to_ix = { ch:i for i, ch in enumerate(chars) }
ix_to_char = { i:ch for i, ch in enumerate(chars) }

#print char_to_ix
#hyperparameters

hidden_size = 100 #number of hidden layers for neurons
seq_length = 25 #number of steps to unroll the RNN
learning_rate = 1e-1

#model parameters
#weight syntax is np.random.randn(target_layer_size, source_layer_size)*.0.01
#weight matrix W_st
#Activation tL = W_st*sL + b_t

#Basic Rnn formulation
#h_t = Wxh Xt + Whh h_t-1 + b_h
#y_t = Why h_t + b_y

Wxh = np.random.randn(hidden_size, vocab_size)*0.01 #input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 #hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 #hidden to output
bh = np.zeros((hidden_size,1)) #hidden bias
by = np.zeros((vocab_size,1)) #output bias

def lossFun(inputs, targets, hprev):
    """
    inputs, targets are both list of integers
    """

    xs, hs, ys, ps = {}, {}, {}, {}
    #set the last element of the hidden state as the hprev
    hs[-1] = np.copy(hprev)
    loss = 0

    #forward pass
    for t in xrange(len(inputs)):
        #one hot encoding of the input
        xs[t] = np.zeros((vocab_size, 1))
        xs[inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))

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
        #softmax over the output
        #Smax(y) = foreach element i .. e^yi/(Sigma_k e^yk)
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
        #loss function for softmax-cross entroy
        #xent(Y,P) = -Sigma(Yi*log(P(i))
        #here Y is the targets vector
        #translating the loss to just -np.log(ps[t][targets[t],0])
        loss += -np.log(ps[t][targets[t],0])

    #backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(xrange(len(inputs))):

        # How to visually think about the back propagation at each stage!
        # for weight derivatives think from the weight's POV (i know it sounds
        # obvious, but it really helps!, trust me)
        # for input derivates think from the POV of all input - ie imagine it
        # affecting all the outputs and as a consequence it's derivative w.r.t to
        # the final output is dependent on the sum of errors in all the immpediate
        # outputs that this neuron touchs, and the chain rules gives us the needed
        # recursive relation ship, so we can only think about the error
        # propagation one stage at a time!

        # ys = Why*hs[t] + by
        # | W11 W12 W13 | * | h1 |  = | W11*h1 + W12*h2 + W13*h3 |
        # | W21 W22 W23 |   | h2 |  = | W21*h1 + W22*h2 + W23*h3 |
        # | W31 W32 W33 |   | h3 |  = | W31*h1 + W32*h2 + W33*h3 |

        # df/dwhy = df/dy * dy/dwhy
        # df/dwhy = df/dy * | h1 h2 h3 |

        # L-1,i-------wij------L,j
        # Lj = Sigma(j,i=0,sizeof(L-1)-1)
        # dwhy = dy * hT
        #      = | y1 | * | h1 h2 h3| <= | y1h1 y1h2 y1h3 |
        #        | y2 |                  | y2h1 y2h2 y2h3 |
        #        | y3 |                  | y3h1 y3h2 y3h3 |

        # error for softmax cross entropy function i.e
        # -np.log(p_target) output function is just p-1
        dy = np.copy(p[t]) 
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + hnext #backprop into h
        dhraw = (1-hs[t]*hs[t]) * dh #backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw) #backprop for the previous step

    #clip to mitigate exploding gradients
    for dparam in [dWhy, dWhh, dWxh, dby, dbh]:
            np.clip(daram, -5, 5, out=dparam) 

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_x, n):
    """
    Sample a sequence from the RNN with starting char seed_x and step n times
    """
    x = np.zeros((vocab_size,1))
    x[seed_x] = 1
    sample_values = []
    for i in xrange(n):
        #update h
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh,h) + bh)
        y = np.dot(Why,h) + by
        p = np.exp(y)/np.sum(np.exp(y))
        #generate a single non-random sample from 0-(vs-1)
        #based on probability distribution p
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size,1))
        x[ix] = 1
        sample_values.append(ix)
        #feed this back to the input
    return sample_values

#Main loop
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), mp.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

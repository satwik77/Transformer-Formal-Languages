## Import relevant libraries and dependencies
import numpy as np
import random
import collections
import math
import torch
from torch.autograd import Variable
from scipy.special import gamma 
from scipy.special import gammaln

class CounterLanguage():
    def __init__(self, num_char):
        self.chars = ['a', 'b', 'c','d']

        # self.vocabulary = vocabulary ## Input vocabulary
        self.vocabulary = ''.join(self.chars[:num_char])
        self.vocab_size = len(self.vocabulary) 

        self.all_letters = self.vocabulary + 'T' ## Output vocabulary (T: termination symbol)
        self.n_letters = len(self.all_letters)

        self.extra_letter = chr(ord(self.vocabulary[-1]) + 1) ## a or b (denoted a/b)
        self.num_char = num_char

    def get_vocab (self):
        return self.vocabulary

    ## Beta-Binomial density (pdf)
    def beta_binom_density(self, alpha, beta, k, n):
        return 1.0*gamma(n+1)*gamma(alpha+k)*gamma(n+beta-k)*gamma(alpha+beta)/ (gamma(k+1)*gamma(n-k+1)*gamma(alpha+beta+n)*gamma(alpha)*gamma(beta))

    ## Beta-Binomial Distribution
    def beta_bin_distrib (self, alpha, beta, N):
        pdf = np.zeros (N+1)

        cumulative = 0.0
        for k in range (N+1):
            prob = self.beta_binom_density (alpha, beta, k, N)
            pdf [k] = prob

        ## Normalize (to fix small precision errors)
        pdf *= (1. / sum(pdf)) 
        return pdf


    def sample_from_a_distrib (self, domain, sample_size, distrib_name):
        N = len(domain)
        if distrib_name == 'uniform':
            return np.random.choice (a=domain, size=sample_size)
        
        elif distrib_name == 'u-shaped':
            alpha = 0.25
            beta = 0.25
            return np.random.choice (a=domain, size=sample_size, p = self.beta_bin_distrib(alpha, beta, N-1))
        
        elif distrib_name == 'right-tailed':
            alpha = 1
            beta = 5
            return np.random.choice (a=domain, size=sample_size, p = self.beta_bin_distrib(alpha, beta, N-1))
        
        elif distrib_name == 'left-tailed':
            alpha = 5
            beta = 1
            return np.random.choice (a=domain, size=sample_size, p = self.beta_bin_distrib(alpha, beta, N-1))
        
        else:
            return Error


    def generate_sample (self, sample_size=1, minv=1, maxv=50, distrib_type='uniform', distrib_display=False):
        input_arr = []
        output_arr = []

        ## domain = [minv, ...., maxv]
        domain = list(range(minv, maxv+1)) 

        nums = self.sample_from_a_distrib (domain, sample_size, distrib_type)

        for num in nums:
            i_seq = ''.join([elt for elt in self.vocabulary for _ in range (num)])
            
            o_seq = ''
            for i in range (self.vocab_size):
                if i == 0:
                    o_seq += self.extra_letter * num ## a or b
                elif i == 1:
                    o_seq += self.vocabulary[i] * (num-1) ## b
                else:
                    o_seq += self.vocabulary[i] * num ## other letters
            o_seq += 'T'  ## termination symbol

            input_arr.append (i_seq)
            output_arr.append (o_seq)

        ## Display the distribution of lengths of the samples
        if distrib_display:
            print ('Distribution of the length of the samples: {}'.format(collections.Counter(nums)))

        return input_arr, output_arr, collections.Counter(nums)


    ## Find letter index from all_letters
    def letterToIndex (self, letter):
        return self.all_letters.find (letter)

    ## Just for demonstration, turn a letter into a <1 x n_letters> tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    ## Turn a line into a <line_length x 1 x n_letters>,
    ## or an array of one-hot letter vectors
    def lineToTensorInput(self, line):
        tensor = torch.zeros(len(line), 1, self.vocab_size)
        for li, letter in enumerate(line):
            if letter in self.all_letters:
                tensor[li][0][self.letterToIndex(letter)] = 1
            else:
                print ('Error 1')
        return tensor

    def lineToTensorOutput(self, line):
        tensor = torch.zeros(len(line), self.n_letters)
        for li, letter in enumerate(line):
            if letter in self.all_letters:
                tensor[li][self.letterToIndex(letter)] = 1
            elif letter == self.extra_letter: # a or b
                tensor[li][self.letterToIndex('a')] = 1
                tensor[li][self.letterToIndex('b')] = 1
            else:
                print ('Error 2')
        return tensor

    def depth_counter(self, seq):
        return len(seq) // self.num_char
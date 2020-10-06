import sys
import numpy as np
import torch
from collections import defaultdict
import ipdb as pdb
import random
import textwrap

sys.setrecursionlimit(5000)

all_pairs = ['()', '[]', '{}', '<>', '+-', 'ab', 'xo']
all_letters = ''
for elt in all_pairs:
	all_letters += elt

init_ascii = 48 ## corresponds to 0

class ShuffleLanguage ():
	def __init__ (self, num_pairs, p, q):
		self.pair_num = num_pairs
		self.pairs = all_pairs [:num_pairs]
		self.vocabulary = all_letters [:2*num_pairs]
		self.n_letters = len (self.vocabulary)

		self.openpar= [elt[0] for elt in self.pairs]
		self.closepar = [elt[1] for elt in self.pairs]

		self.p = p
		self.q = q
	
	# returns the vocabulary
	def return_vocab (self):
		return self.vocabulary

	# generate a sample
	# def generate (self, current_size, max_size):
	# 	# Houston, we have a problem here. (Limit exceeded.)
	# 	if current_size >= max_size: 
	# 		return ''
		
	# 	prob = random.random()
	# 	# Grammar: S -> (_i S )_i with prob p | SS with prob q | empty with prob 1 - (p+q)
	# 	if prob < self.p:
	# 		chosen_pair = np.random.choice (self.pairs) # randomly pick one of the pairs.
	# 		sample = chosen_pair[0] + self.generate (current_size + 2, max_size) + chosen_pair [1]
	# 		if len (sample) <= max_size:
	# 			return sample
	# 	elif prob < self.p + self.q:
	# 		sample = self.generate (current_size, max_size) + self.generate (current_size, max_size)
	# 		if len (sample) <= max_size:
	# 			return sample
	# 	else:
	# 		return ''

	# 	return ''
	

	def generate (self, current_size, max_size, max_depth):
		# Houston, we have a problem here. (Limit exceeded.)
		if current_size >= max_size: 
			return ''
		
		prob = random.random()
		# Grammar: S -> (_i S )_i with prob p | SS with prob q | empty with prob 1 - (p+q)
		if prob < self.p:
			chosen_pair = np.random.choice (self.pairs) # randomly pick one of the pairs.
			sample = chosen_pair[0] + self.generate (current_size + 2, max_size, max_depth) + chosen_pair [1]
			if len (sample) <= max_size:
				if (max_depth == -1) or (len(sample)==0) or (self.depth_counter(sample).sum(1).max() <= max_depth):
					return sample
		elif prob < self.p + self.q:
			sample = self.generate (current_size, max_size, max_depth) + self.generate (current_size, max_size, max_depth)
			if len (sample) <= max_size:
				if (max_depth == -1) or (len(sample)==0) or (self.depth_counter(sample).sum(1).max() <= max_depth):
					return sample
		else:
			return ''

		return ''

	# # generate 'num' number of samples
	# def generate_list (self, num, min_size, max_size):
	# 	arr = []
	# 	size_info = defaultdict (list)
	# 	counter = 0
	# 	while counter < num:
	# 		sample = self.generate (0, max_size)
	# 		if sample not in arr and len(sample) >= min_size:
	# 			counter += 1
	# 			arr.append (sample)
	# 			size_info [len(sample)].append(sample)
	# 			if counter % 500 == 0:
	# 				print ('{} samples generated.'.format(counter))
				
	# 	return arr, size_info



	def generate_list (self, num, min_size, max_size, min_depth=0, max_depth=-1):
		arr = []
		size_info = defaultdict (list)
		counter = 0
		while counter < num:
			sample = self.generate (0, max_size, max_depth)
			if sample not in arr and len(sample) >= min_size:
				if (max_depth==-1) or (self.depth_counter(sample).sum(1).max() >= min_depth):
					counter += 1
					arr.append (sample)
					size_info [len(sample)].append(sample)
					print ('{}/{} samples generated.'.format(counter, num), end = '\r', flush = True)
		print()
		return arr, size_info



	def output_generator (self, seq):
		output_seq = ''
		depths = self.depth_counter(seq)
		for i in range(len(seq)):
			for j in range(self.pair_num):
				if depths[i][j] == 0:
					output_seq += '0'
				else:
					output_seq += '1'
		return output_seq


	def depth_counter (self, seq):
		dyck_counter = np.zeros (self.pair_num)
		max_depth = np.zeros ((len(seq), self.pair_num))
		counter = 0
		for elt in seq:
			indexl = 0
			if elt in self.openpar:
				indexl = self.openpar.index(elt)
				dyck_counter[indexl] += 1
			else:
				indexl = self.closepar.index(elt)
				dyck_counter[indexl] -= 1
			max_depth[counter] = dyck_counter
			counter += 1
		max_depth = max_depth#.sum(1).max()
		return max_depth

	def training_set_generator (self, num, min_size, max_size, min_depth=0, max_depth= -1):
		input_arr, input_size_arr = self.generate_list (num, min_size, max_size, min_depth, max_depth)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr, input_size_arr

	def lineToTensorOutput(self, line):
		assert len(line) % self.pair_num == 0
		line_parts = textwrap.wrap(line, self.pair_num)

		tensor = torch.ones(len(line)//self.pair_num, self.n_letters)
		for i,part in enumerate(line_parts):
			for j,char in enumerate(part):
				tensor[i, 2*j + 1] = float(char)
		return tensor
		

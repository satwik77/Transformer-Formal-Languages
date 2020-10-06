import numpy as np
import torch
from abc import ABC, abstractmethod 
import ipdb as pdb
letters = ['a','b','c','d','e','f','g','h']

class NonStarFreeLanguage(object):

	def __init__(self, n):
		self.sigma = letters[:n]
		self.char2id = {ch:i for i,ch in enumerate(self.sigma)}
		self.n_letters = n

	@abstractmethod
	def belongToLang(self, seq):
		pass

	@abstractmethod
	def generate_string(self, min_length, max_length):
		pass

	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(min_length, max_length)
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	
	def output_generator(self, seq):
		output_seq = ''
		for i in range(1, len(seq)+1):
			part_seq = seq[:i]
			if self.belongToLang(part_seq):
				output_seq += '1'
			else:
				output_seq += '0'
		return output_seq

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr
	
	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), 2)
		for li, letter in enumerate(line):
			tensor[li][int(letter)] = 1.0
		return tensor

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))


class ABABStarLanguage(NonStarFreeLanguage):

	def __init__(self, n = 2):
		super(ABABStarLanguage, self).__init__(n)
	
	def belongToLang(self, seq):
		sublen = self.n_letters * 2

		if len(seq) % sublen != 0:
			return False

		for i in range(0, len(seq), sublen):
			subseq = seq[i:i+sublen]
			if subseq != ''.join(self.sigma + self.sigma):
				return False

		return True

	def generate_string(self, min_length, max_length):
		sublen = self.n_letters * 2
		num_ababs = (min_length + np.random.randint(max_length - min_length + 1))//sublen
		string = ''.join([''.join(self.sigma+self.sigma) for _ in range(num_ababs)])
		return string

class AAStarLanguage(NonStarFreeLanguage):

	def __init__(self, n):
		super(AAStarLanguage, self).__init__(n = 1)
		self.n = n

	def belongToLang(self, seq):
		req_subseq = ''.join([self.sigma[0] for _ in range(self.n)])
		sublen = len(req_subseq)

		if len(seq) % sublen != 0:
			return False

		for i in range(0, len(seq), sublen):
			subseq = seq[i:i+sublen]
			if subseq != req_subseq:
				return False
		
		return True

	def generate_string(self, min_length, max_length):
		req_subseq = ''.join([self.sigma[0] for _ in range(self.n)])
		sublen = len(req_subseq)
		num_aas = (min_length + np.random.randint(max_length - min_length + 1))//sublen
		string = ''.join([''.join(req_subseq) for _ in range(num_aas)])
		return string	

class AnStarA2Language(NonStarFreeLanguage):

	def __init__(self, n):
		super(AnStarA2Language, self).__init__(n = 1)
		self.n = n
		self.lang = AAStarLanguage(n)

	def generate_string(self, min_length, max_length):
		string =  self.lang.generate_string(min_length, max_length) + 'aa'
		return string

	def belongToLang(self, seq):
		if len(seq) < 2:
			return False
		if seq[-2:] != 'aa':
			return False
		return self.lang.belongToLang(seq[:-2])
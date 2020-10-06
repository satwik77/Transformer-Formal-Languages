import numpy as np
import torch
from collections import defaultdict
#ops = ['~', '^', '+', '*', '<', '>']
ops = ['~', '+', '>']
symb2nary = {
				'0' : 0,
				'1' : 0,
				'~' : 1,
				'+' : 2,
				'>' : 3
			}

class NAryBooleanExpLang(object):

	def __init__(self, n = 3, p = 0.5):
		assert n <= 3
		self.n = n
		self.p = p
		self.sigma = ['0', '1']
		self.expr = [exp for exp,m in symb2nary.items() if m <= n and m!=0]
		self.nary2symbs = defaultdict(list)
		for exp,m in symb2nary.items():
			self.nary2symbs[m].append(exp)

	def expand_expr(self):
		choices = [0] + [i for i in range(1,self.n+1)]
		ps = [1-self.p] + [self.p/self.n for i in range(self.n)]
		toss = np.random.choice(choices, p = ps)
		if toss == 0:
			return np.random.choice(['0', '1']), toss
		else:
			opr = np.random.choice(self.nary2symbs[toss])
			return opr, toss


	def generate_string(self, maxlength):
		count = 1
		expr = ''
		while len(expr) < maxlength:
			symb, new_exprs = self.expand_expr()
			expr += symb
			count = count - 1 + new_exprs
			if count == 0:
				return expr
		return ''

	def generate_list(self, num, min_size, max_size):
		arr = set()
		while len(arr) < num:
			expr = self.generate_string(max_size)
			if len(expr) < min_size:
				continue
			if expr == '' or expr in arr:
				continue
			arr.add(expr)
			print("Generated {}/{} expressions".format(len(arr), num), end = '\r', flush = True)
		print()
		return list(arr)


	def output_generator(self, seq):
		out_seq = ['0' for i in range(len(seq) - 1)] + ['1']
		return ''.join(out_seq)

	def training_set_generator(self, num, min_size, max_size):
		input_seq = self.generate_list(num, min_size, max_size)
		output_seq = []
		for i in range(len(input_seq)):
			output_seq.append(self.output_generator(input_seq[i]))
		return input_seq, output_seq

	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), 2)
		for li, letter in enumerate(line):
			tensor[li][int(letter)] = 1.0
		return tensor

	def depth_counter(self, seq):
		count = 0
		depths = []
		for ch in seq:
			count = count + symb2nary[ch] - 1
			depths.append(count)
		return np.array(depths)[:, np.newaxis]

import numpy as np
import torch
import ipdb as pdb
import textwrap

letters = ['a','b','c','d','e','f','g','h']

def get_sigma_star(sigma, length):
	choices = sigma
	string = ''
	while len(string) < length:
		symbol = np.random.choice(choices)
		string += symbol
	return string

class AAStarBBStarLanguage(object):

	def __init__(self, n = 5):
		self.sigma = letters[:n] + ['T']
		self.char2id = {ch:i for i,ch in enumerate(self.sigma)}
		self.n_letters = n + 1
		'''
		self.cond_probs = np.zeros(n, n + 1)
		for i in range(n):
			self.cond_probs[i,i] = 0.5
			self.cond_probs[i,i+1] = 0.5
		'''

	def generate_string(self, min_length, max_length):
		'''
		string = 'a'
		curr_char = 'a'
		symbols = self.sigma + 'T'
		while True:
			curr_char = np.random.choice(symbols, p = self.cond_probs[self.char2id[curr_char]]
			if curr_char == 'T':
				break
			string += curr_char

		return string
		'''
		string = ''
		delta_len = max_length - min_length + 1
		total_count = delta_len
		for symbol in self.sigma[:-1]:
			if total_count > 0:
				count = np.random.randint(total_count) + 1
			else:
				count = 0
			symb_count = min_length//(self.n_letters-1) + count
			string += ''.join([symbol for i in range(symb_count)]) 
			total_count = total_count - count
		string += 'T'
		return string

	def generate_list(self, num, min_length, max_length):
		'''
		arr = []
		for l in range(min_length, max_length + 1):
			strings = []
			self.generate_strings(l, w = '', strings = strings)
			arr += strings
		arr = list(np.random.choice(arr, size = min(num, len(arr)), replace = False))

		return arr
		'''
		arr = []
		while len(arr) < num:
			string = self.generate_string(min_length, max_length)
			if string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				if self.sigma[-2] in string:
					arr.append(string)
					print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	
	def output_generator(self, seq):
		output_seq = ''
		for symbol in seq:
			if symbol == 'T':
				break
			output_seq += self.sigma[self.char2id[symbol] + 1]
		return output_seq

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr
	
	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line)+1, self.n_letters)
		for li, letter in enumerate(line):
			letter_id = self.char2id[letter]
			tensor[li][letter_id - 1] = 1.0
			tensor[li][letter_id] = 1.0
		return tensor

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class ABStarLanguage(object):

	def __init__(self, n):
		assert n == 2
		self.sigma = letters[:2]
		self.char2id = {ch:i for i,ch in enumerate(self.sigma)}
		self.n_letters = n

	def belongToLang(self, seq):
		sublen = self.n_letters

		if len(seq) % sublen != 0:
			return False

		for i in range(0, len(seq), sublen):
			subseq = seq[i:i+sublen]
			if subseq != ''.join(self.sigma):
				return False

		return True

	def generate_string(self, min_length, max_length):
		sublen = self.n_letters
		num_ababs = (min_length + np.random.randint(max_length - min_length + 1))//sublen
		string = ''.join([''.join(self.sigma) for _ in range(num_ababs)])
		return string

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

class CStarAnCStarLanguage(object):

	def __init__(self, n, sigma = ['a', 'c'], a = 'a'):
		self.n = n
		self.sigma = sigma
		self.a = a
		self.n_letters = 2

	def generate_string(self, maxlength):

		#Get c's for the left part. Can be from 0 to maxlength - 1
		num_lcs = np.random.randint(0, maxlength - self.n + 1)
		#Get c's for the right part. Can be from 0 to (maxlength - left c's - 1)
		num_rcs = np.random.randint(0, maxlength - num_lcs)

		string = 'c'*num_lcs + self.a*self.n + 'c'*num_rcs

		return string
	
	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def output_generator(self, seq):
		num_as_found = 0
		out_seq = ''

		for i,ch in enumerate(seq):
			if ch == 'c' and num_as_found == 0:
				out_seq += 'b'
			elif ch == 'c' and num_as_found > 0:
				out_seq += 'c'
			elif ch == 'a' and num_as_found < self.n - 1:
				out_seq += 'a'
				num_as_found += 1
			else:
				out_seq += 'c'
				num_as_found += 1

		return out_seq

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), 2)
		for i,ch in enumerate(line):
			if ch == 'a':
				tensor[i][0] = 1.0
				tensor[i][1] = 0.0
			elif ch == 'b':
				tensor[i][0] = 1.0
				tensor[i][1] = 1.0
			else:
				tensor[i][0] = 0.0
				tensor[i][1] = 1.0

		return tensor

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class CStarAnCStarBnCStarLanguage(object):

	def __init__(self, n):
		self.n = n
		self.sigma = ['a','b','c']
		self.n_letters = 3
		self.lang1 = CStarAnCStarLanguage(n, ['a','c'], a = 'a')
		self.lang2 = CStarAnCStarLanguage(n, ['b','c'], a = 'b')

	def generate_string(self, maxlength):

		string1 = self.lang1.generate_string(maxlength//2)
		string2 = self.lang2.generate_string(maxlength//2) 
		string  = string1 + string2
		return string

	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def output_generator(self, seq):
		output_seq = ''
		num_as = 0
		num_bs = 0
		for i,ch in enumerate(seq):
			if ch == 'c' and num_as == 0:
				output_seq += 'd'
			elif ch == 'c' and num_as == self.n and num_bs != self.n:
				output_seq += 'e'
			elif ch == 'c' and num_bs == self.n:
				output_seq += 'c'
			elif ch == 'a' and num_as < self.n - 1:
				output_seq += 'a'
				num_as += 1
			elif ch == 'a' and num_as == self.n-1:
				output_seq += 'e'
				num_as += 1
			elif ch == 'b' and num_bs < self.n - 1:
				output_seq += 'b'
				num_bs += 1
			elif ch == 'b' and num_bs == self.n-1:
				output_seq += 'c'
				num_bs += 1
			else:
				pdb.set_trace()

		return output_seq

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), 3)
		for i,ch in enumerate(line):
			if ch == 'a':
				tensor[i][0] = 1.0
			elif ch == 'b':
				tensor[i][1] = 1.0
			elif ch == 'c':
				tensor[i][2] = 1.0
			elif ch == 'd':
				tensor[i][0] = 1.0
				tensor[i][2] = 1.0
			else:
				tensor[i][1] = 1.0
				tensor[i][2] = 1.0

		return tensor

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class CStarAnCStarv2Language(object):
	'''
	Generates the strings belonging to the language:
		c*a^nc*bc*a^nc*
	'''
	
	def __init__(self, n = 1):
		self.n = n
		self.sigma = ['a', 'b','c']
		self.n_letters = 3
		self.lang = CStarAnCStarLanguage(n)

	def generate_string(self, maxlength):
		string1 = self.lang.generate_string(maxlength//2)
		string2 = self.lang.generate_string(maxlength//2)

		string = string1 + 'b' + string2

		return string

	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def output_generator(self, seq):
		output_seq = ''
		num_as = 0
		found_b = False
		for i,ch in enumerate(seq):
			# if the encountered c is in [c*]a^nc*bc*a^nc*
			if ch == 'c' and num_as == 0 and not found_b:
				output_seq += 'd' # d means a+c are allowed
			
			#if the encountered c is in c*a^n[c*]bc*a^nc*
			elif ch == 'c' and num_as == self.n and not found_b:
				output_seq += 'e' # e means b+c are allowed

			#if the encountered c is in c*a^nc*b[c*]a^nc*
			elif ch == 'c' and num_as == 0 and found_b:
				output_seq += 'd'
			
			#if the encountered c is in c*a^n[c*]bc*a^n[c*]
			elif ch == 'c' and num_as == self.n and found_b:
				output_seq += 'c'

			#if the encountered a is in c*[a^n-1]a[c*]bc*a^nc*
			elif ch == 'a' and num_as < self.n - 1 and not found_b:
				output_seq += 'a'
				num_as += 1

			#if the encountered a is in c*a^n-1[a][c*]bc*a^nc*
			elif ch == 'a' and num_as == self.n - 1 and not found_b:
				output_seq += 'e'
				num_as += 1
			
			#if the encountered a is in c*a^n[c*]bc*[a^n-1]ac*
			elif ch == 'a' and num_as < self.n - 1 and found_b:
				output_seq += 'a'
				num_as += 1

			#if the encountered a is in c*a^n[c*]bc*a^n-1[a]c*
			elif ch == 'a' and num_as == self.n - 1 and found_b:
				output_seq += 'c'
				num_as += 1

			#if we encounter b
			elif ch == 'b':
				output_seq += 'd'
				found_b = True
				num_as = 0

			#Sanity Check
			else:
				pdb.set_trace()

		return output_seq

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), 3)
		for i,ch in enumerate(line):
			if ch == 'a':
				tensor[i][0] = 1.0
			elif ch == 'b':
				tensor[i][1] = 1.0
			elif ch == 'c':
				tensor[i][2] = 1.0
			elif ch == 'd':
				tensor[i][0] = 1.0
				tensor[i][2] = 1.0
			else:
				tensor[i][1] = 1.0
				tensor[i][2] = 1.0

		return tensor

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))


class CABLanguage(object):
	'''
	Regular expression: c(a|b)*
	'''
	
	def __init__(self):
		self.sigma = ['c', 'a', 'b']
		self.n_letters = len(self.sigma)

	def generate_string(self, max_length):
		a_or_b = np.random.choice(['a', 'b'])
		length = np.random.randint(0, max_length)
		string = 'c' + get_sigma_star(sigma = ['a', 'b'], length = length)
		return string
	
	def output_generator(self, seq):
		out_arr = []
		for ch in seq:
			out_arr.append(['a','b'])

		return out_arr 


class ABDLanguage(object):

	'''
	Regular expression: abd(a|b|d)*
	'''

	def __init__(self):
		self.sigma = ['a', 'b','d']
		self.n_letters = len(self.sigma)

	def generate_string(self, max_length):
		a_or_b_or_d = np.random.choice(['a','b','d'])
		length = np.random.randint(0, max_length - 2)
		string = 'abd' + get_sigma_star(sigma = ['a', 'b', 'd'], length = length)

		return string

	def output_generator(self, seq):
		out_arr = [['b'], ['d'], ['a', 'b', 'd']]
		for ch in seq[3:]:
			out_arr.append(['a', 'b', 'd'])

class CAB_n_ABDLanguage(object):
	
	'''
	Regular expression: c(a|b)*abd(a|b|d)*
	'''

	def __init__(self):
		self.sigma = ['a', 'b', 'c', 'd']
		self.n_letters = len(self.sigma)
		self.lang1 = CABLanguage()
		self.lang2 = ABDLanguage()

	def generate_string(self, max_length):

		str1 = self.lang1.generate_string(max_length - 3)
		str2 = self.lang2.generate_string(max_length - len(str1))
		string = str1 + str2

		return string

	def output_generator(self, seq):
		abd_idx = seq.find('abd')
		substr1 = seq[:abd_idx]
		substr2 = seq[abd_idx:]

		out_seq = ''
		last_char = ''

		for ch in substr1:
			if ch == 'c':
				out_seq += '1100'
			elif ch == 'a':
				out_seq += '1100'
			elif ch == 'b':
				if last_char == 'a':
					out_seq += '1101'
				else:
					out_seq += '1100'
			last_char = ch

		out_seq += '1100'
		out_seq += '1101'
		out_seq += '1101'
		for ch in substr2[3:]:
			out_seq += '1101'
		return out_seq

	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		assert len(line) % 4 == 0
		parts = textwrap.wrap(line, 4)
		tensor = []
		for i,part in enumerate(parts):
			tensor.append(list(map(float, list(part))))

		tensor = torch.tensor(tensor)
		return tensor

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class D_nLanguage(object):

	def __init__(self, n):
		self.sigma = ['a','b']
		self.n_letters = 2
		self.n = n

	def random_select_length(self, maxlength, mean_ratio = 0.75, std_ratio = 0.1):
		mean = maxlength * mean_ratio
		std = std_ratio * mean

		length = int(std * np.random.randn() + mean)
		return length

	def generate_d_n(self, n, maxlength):
		if n == 0:
			return ''
		if maxlength == 0:
			return ''

		d_n = ''
		while len(d_n) < maxlength:
			length_d_n_min_1 = self.random_select_length(maxlength)
			d_n_min_1 = self.generate_d_n(n-1, length_d_n_min_1)
			d_n += 'a{}b'.format(d_n_min_1)

		return d_n


	def generate_string(self, maxlength):
		length = self.random_select_length(maxlength)
		string = self.generate_d_n(self.n, length)

		return string

	def find_depth(self, seq):
		counter = {'a' : 1, 'b': -1}
		depth = 0
		for i,ch in enumerate(seq):
			depth += counter[ch]
		return depth

	def get_final_state(self, seq):
		depth = self.find_depth(seq)
		if depth == 0:
			return 'q_0'
		elif depth == self.n:
			return 'q_n'
		else:
			return 'q_i'

	def output_generator(self, seq):
		output_seq = ''
		for i in range(len(seq)):
			q = self.get_final_state(seq[:i+1])
			if q == 'q_0':
				output_seq += '10'
			elif q == 'q_i':
				output_seq += '11'
			else:
				output_seq += '01'

		return output_seq

	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		assert len(line) % 2 == 0
		parts = textwrap.wrap(line, 2)
		tensor = []
		for i,part in enumerate(parts):
			tensor.append(list(map(float, list(part))))

		tensor = torch.tensor(tensor)
		return tensor

	def depth_counter(self, seq):
		depths = np.zeros((len(seq), 1))
		for i in range(len(seq)):
			depths[i] = self.find_depth(seq[:i+1])
		return depths
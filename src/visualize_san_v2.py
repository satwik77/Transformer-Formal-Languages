import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.dyck_generator import DyckLanguage
from src.utils.helper import *
from src.utils.logger import get_logger
from src.utils.sentence_processing import sents_to_idx
from src.components.transformers import TransformerModel
import pickle
from attrdict import AttrDict
import logging
import os
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import collections
from functools import partial
import ipdb as pdb

all_pairs = ['()', '[]', '{}', '<>', '+-', 'ab', 'xo']
all_open = ['(', '[', '{', '<','+','a','x']
def generate_visualizations(model, config, voc, src_str = '((()))(())()', run_name = 'SAN', iteration = 0,score = 1.0, device = 'cuda:1'):
	
	dlang = DyckLanguage(config.num_par, 0.5, 0.25)
	
	#Convert source string to ids tensor
	src = sents_to_idx(voc, [src_str]).transpose(0,1).to(device)[:-1]

	#Create directory to save visualizations
	dir_path = os.path.join("Figures", run_name)
	if os.path.exists(dir_path) == False:
		os.mkdir(dir_path)
	
	# Ploting attention weights
	def visualize_attn(src, src_str):
		output, attn = model.model(src, get_attns = True)
		src_len = len(src_str)
		attn_maps = []
		attn_map = attn[0][0,:src_len,:src_len].detach().cpu().numpy()
		for i in range(config.depth):
			for j in range(config.heads):
				attn_map = attn[i][0,j,:src_len,:src_len].detach().cpu().numpy()
				plt.figure(figsize=  (15,10))
				sns.set(font_scale = 1.5)
				g = sns.heatmap(np.log(attn_map), mask = (attn_map == 0).astype(float),annot=attn_map, cmap = sns.cubehelix_palette(100, start=0.7, rot=-0.5, gamma = 1.5), vmin = -2.6, vmax = 0, cbar = False, xticklabels=list(src_str), yticklabels=list(src_str), linewidths=.5) #cmap="YlGnBu")
				yticks = g.set_yticklabels(labels = list(src_str), rotation = 360, size = 30)
				xticks = g.set_xticklabels(labels = list(src_str), size = 30)
				plt.title('Attention Weights Layer: {} Head: {} (It :{})'.format(i+1, j+1, iteration), size = 20)
				fig = g.get_figure()
				fig.savefig(os.path.join(dir_path, 'attn_weights_depth-{}_heads-{}_it-{}.png'.format(i+1, j+1, iteration)), bbox_inches='tight')
				attn_maps.append(attn_map)
		return attn_maps

	attn_maps = visualize_attn(src, src_str)

	# Computing and Ploting Intermediate representations

	# Obtaining Embeddings
	embeddings = model.model.encoder(src) * np.sqrt(config.d_model)
	embed_unq = torch.unique(embeddings, dim = 0)
	embed_unq = embed_unq.detach().cpu().numpy().squeeze()
	#Plotting Embeddings
	plt.figure(figsize = (10, 3))
	#embed_unq = embeddings_np[[0,3]]
	g = sns.heatmap(embed_unq, annot = embed_unq,  cmap = sns.color_palette("coolwarm", 7),
					linewidth=1.5, linecolor = 'black', yticklabels = ['(', ')', '[',']'])
	g.set_title('Embeddings (It: {})'.format(iteration))
	fig = g.get_figure()
	fig.savefig(os.path.join(dir_path, 'embeddings_it-{}.png'.format(iteration)), bbox_inches='tight')
	
	# Computing queries, keys and values
	kqv = list(model.model.transformer_encoder.parameters())[0].detach()
	b = list(model.model.transformer_encoder.parameters())[1].detach()
	query_matrix, query_bias = kqv[:config.d_model], b[:config.d_model]
	key_matrix, key_bias = kqv[config.d_model:config.d_model * 2], b[config.d_model:config.d_model * 2]
	value_matrix, value_bias = kqv[config.d_model * 2:], b[config.d_model * 2:]
	kqv_vectors = torch.mm(embeddings.squeeze(), kqv.transpose(0,1)) + b
	queries, keys, values = kqv_vectors[:,:config.d_model], kqv_vectors[:,config.d_model:config.d_model * 2], kqv_vectors[:,config.d_model * 2:]
	# Plotting Query Matrix
	sns.set(font_scale = 1.2)
	query_matrix_np = query_matrix.detach().cpu().numpy().squeeze()
	query_bias_np = query_bias.detach().cpu().numpy().squeeze()[:,None]
	f,(ax1,ax2, ax3) = plt.subplots(1,3,sharey=False, gridspec_kw={'width_ratios':[6,2, 0.8]}, figsize = (10,5))
	#f.tight_layout() 
	g1 = sns.heatmap(query_matrix_np,annot = query_matrix_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax1, cbar = False)
	g1.set_title('Query Matrix (It: {})'.format(iteration))

	g2 = sns.heatmap(query_bias_np, annot = query_bias_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax2, cbar_ax = ax3)
	g2.set_title('Query Bias (It: {})'.format(iteration))
	f.savefig(os.path.join(dir_path, 'query_wb_it-{}.png'.format(iteration)), bbox_inches='tight')
	# Plotting Key Matrix
	sns.set(font_scale = 1.2)
	key_matrix_np = key_matrix.detach().cpu().numpy().squeeze()
	key_bias_np = key_bias.detach().cpu().numpy().squeeze()[:,None]
	#plt.figure(figsize = (10, 10))
	f,(ax1,ax2, ax3) = plt.subplots(1,3,sharey=False, gridspec_kw={'width_ratios':[6,2, 0.8]}, figsize = (10,5))
	#f.tight_layout() 
	g1 = sns.heatmap(key_matrix_np,annot = key_matrix_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax1, cbar = False)
	g1.set_title('Key Matrix (It: {})'.format(iteration))

	g2 = sns.heatmap(key_bias_np, annot = key_bias_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax2, cbar_ax = ax3)
	g2.set_title('Key Bias (It: {})'.format(iteration))
	f.savefig(os.path.join(dir_path, 'key_wb_it-{}.png'.format(iteration)), bbox_inches='tight')
	# Ploting Value Matrix
	sns.set(font_scale = 1.2)
	value_matrix_np = value_matrix.detach().cpu().numpy().squeeze()
	value_bias_np = value_bias.detach().cpu().numpy().squeeze()[:,None]
	#plt.figure(figsize = (10, 10))
	f,(ax1,ax2, ax3) = plt.subplots(1,3,sharey=False, gridspec_kw={'width_ratios':[6,2, 0.8]}, figsize = (10,5))
	#f.tight_layout() 
	g1 = sns.heatmap(value_matrix_np,annot = value_matrix_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax1, cbar = False)
	g1.set_title('Value Matrix (It: {})'.format(iteration))

	g2 = sns.heatmap(value_bias_np, annot = value_bias_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax2, cbar_ax = ax3)
	g2.set_title('Value Bias (It: {})'.format(iteration))
	f.savefig(os.path.join(dir_path, 'value_wb_it-{}.png'.format(iteration)), bbox_inches='tight')
	#Plotting value vectors
	plt.figure(figsize = (10, 3))
	values_unq = torch.unique(values, dim = 0)
	values_unq = values_unq.detach().cpu().numpy()
	#pdb.set_trace()
	g = sns.heatmap(values_unq, annot = values_unq,  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', yticklabels = ['(', ')', '[',']'])
	g.set_title('Values (It: {})'.format(iteration))
	fig = g.get_figure()
	fig.savefig(os.path.join(dir_path, 'values_it-{}.png'.format(iteration)), bbox_inches='tight')
	
	# Computing Attention Map
	n = len(queries)
	mask = torch.tensor(np.triu(np.ones((n,n))).T).float().to(device)
	scores = torch.mm(queries, keys.T) / (np.sqrt(3))
	scores = scores * mask + (-1e9) * (1 - mask)
	attn_map = nn.functional.softmax(scores, dim = -1)
	#pdb.set_trace()
	#assert np.allclose(attn_map.detach().cpu().numpy(), attn_maps[0])
	
	# Computing attention outputs
	attn_outs = torch.mm(attn_map.float(), values.float())
	# Ploting attention outputs
	seq = src_str
	depths = dlang.depth_counter(seq).squeeze()
	lens = np.array([i+1 for i in range(len(seq))])
	dlratios = [depths[:,i]/lens for i in range(depths.shape[1])]
	sns.set(font_scale = 3 ,style = 'ticks', rc={"lines.linewidth": 4})
	src_chars = src_str
	src_charsv0 = list(src_chars)
	src_chars = ['{}_{}'.format(ch,i) for i,ch in enumerate(src_chars)]
	attn_values = attn_outs.detach().cpu().numpy()
	data = pd.DataFrame([src_chars, attn_values]).transpose()
	data.columns = ['dyck', '0-Element']
	fig = plt.figure(figsize = (25, 10))
	plt.plot(src_chars, attn_values[:,0], marker = 'o', label = 'Coordinate-0', markersize = 12, color = 'r')
	plt.plot(src_chars, attn_values[:,1], marker = 'D', label = 'Coordinate-1', markersize = 12, color = 'm')
	plt.plot(src_chars, attn_values[:,2], marker = 'v', label = 'Coordinate-2', markersize = 12, color = 'g')
	for i,dlratio in enumerate(dlratios):
		plt.plot(src_chars, dlratio,'--', marker = 's', markersize = 12, color = 'c', label = '{} DL Ratio'.format(all_open[i]))
	plt.legend(loc="upper right")
	plt.title("Output of Self-Attention Block (It: {})".format(iteration))
	plt.grid()
	plt.rc('grid', linestyle="-", color='black')
	plt.savefig(os.path.join(dir_path, 'attn_outs-{}.png'.format(iteration)), bbox_inches='tight')

	# Computing outputs on applying a linear layer on attention outputs
	attn_ffn_w = list(model.model.transformer_encoder.parameters())[2].detach()
	attn_ffn_b = list(model.model.transformer_encoder.parameters())[3].detach()
	attn_ffn = torch.mm(attn_outs, attn_ffn_w.transpose(0,1)) + attn_ffn_b
	sns.set(font_scale = 1.2)
	attn_ffn_w_np = attn_ffn_w.detach().cpu().numpy().squeeze()
	attn_ffn_b_np = attn_ffn_b.detach().cpu().numpy().squeeze()[:,None]
	#plt.figure(figsize = (10, 10))
	f,(ax1,ax2, ax3) = plt.subplots(1,3,sharey=False, gridspec_kw={'width_ratios':[6,2, 0.8]}, figsize = (10,5))
	f.tight_layout() 
	g1 = sns.heatmap(attn_ffn_w_np,annot = attn_ffn_w_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax1, cbar = False)
	g1.set_title('Attn-FFN Matrix (It: {})'.format(iteration))

	g2 = sns.heatmap(attn_ffn_b_np, annot = attn_ffn_b_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax2, cbar_ax = ax3)
	g2.set_title('Attn-FFN Bias (It: {}))'.format(iteration))
	f.savefig(os.path.join(dir_path, 'attnffn_wb_it-{}.png'.format(iteration)), bbox_inches='tight')

	# Last few important layers
	ln1 = model.model.transformer_encoder.layers[0].norm1
	ln2 = model.model.transformer_encoder.layers[0].norm2
	linear1 = model.model.transformer_encoder.layers[0].linear1
	linear2 = model.model.transformer_encoder.layers[0].linear2
	try:
		activation = model.model.transformer_encoder.layers[0].activation
	except:
		activation = F.relu
	# Feeding attn_ffn obtained in the last cell to residual and layer norm layers
	res_out = embeddings.squeeze() + attn_ffn
	res_ln_out = ln1(res_out)

	# Applying a feed forward network (d_model -> d_ffn -> d_model) to the resulting output from last set
	pos_ffn = linear2(activation(linear1(res_ln_out)))

	# Applying residual + layer norm to the vectors obtained from last step
	res_out2 = (res_ln_out + pos_ffn)
	res_ln_out2 = ln2(res_out2)
	
	pos_ffn1_w = list(linear1.parameters())[0]
	pos_ffn1_b = list(linear1.parameters())[1]

	#Plotting Pos_FFN-1 Weights
	sns.set(font_scale = 1.2)
	pos_ffn1_w_np = pos_ffn1_w.detach().cpu().numpy().squeeze()
	pos_ffn1_b_np = pos_ffn1_b.detach().cpu().numpy().squeeze()[:,None]
	#plt.figure(figsize = (10, 10))
	f,(ax1,ax2, ax3) = plt.subplots(1,3,sharey=False, gridspec_kw={'width_ratios':[6,2, 0.4]}, figsize = (10,5))
	#f.tight_layout() 
	g1 = sns.heatmap(pos_ffn1_w_np,annot = pos_ffn1_w_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax1, cbar = False)
	g1.set_title('Pos-FFN Layer-1 Matrix (It: {})'.format(iteration))

	g2 = sns.heatmap(pos_ffn1_b_np, annot = pos_ffn1_b_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax2, cbar_ax = ax3)
	g2.set_title('Pos-FFN Layer-1 Bias (It: {})'.format(iteration))
	f.savefig(os.path.join(dir_path, 'posffn1_wb_it-{}.png'.format(iteration)), bbox_inches='tight')

	pos_ffn2_w = list(linear2.parameters())[0]
	pos_ffn2_b = list(linear2.parameters())[1]

	#Plotting Pos_FFN Weights
	sns.set(font_scale = 1.2)
	pos_ffn2_w_np = pos_ffn2_w.detach().cpu().numpy().squeeze()
	pos_ffn2_b_np = pos_ffn2_b.detach().cpu().numpy().squeeze()[:,None]
	#plt.figure(figsize = (10, 10))
	f,(ax1,ax2, ax3) = plt.subplots(1,3,sharey=False, gridspec_kw={'width_ratios':[6,2, 0.4]}, figsize = (10,5))
	#f.tight_layout() 
	g1 = sns.heatmap(pos_ffn2_w_np,annot = pos_ffn2_w_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax1, cbar = False)
	g1.set_title('Pos-FFN Layer-1 Matrix (It: {})'.format(iteration))

	g2 = sns.heatmap(pos_ffn2_b_np, annot = pos_ffn2_b_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', ax = ax2, cbar_ax = ax3)
	g2.set_title('Pos-FFN Layer-2 Bias (It: {})'.format(iteration))
	f.savefig(os.path.join(dir_path, 'posffn2_wb_it-{}.png'.format(iteration)), bbox_inches='tight')

	# Feeding the encoder representations (obtained above) to the output linear layer (called decoder)
	decoder_weights = list(model.model.decoder.parameters())[0].detach()
	decoder_reps = torch.mm(res_ln_out2, decoder_weights.T)
	sns.set(font_scale = 1.2)
	decoder_w_np = decoder_weights.detach().cpu().numpy().squeeze()
	plt.figure(figsize = (10, 5))
	#f,(ax1,ax2, ax3) = plt.subplots(1,3,sharey=False, gridspec_kw={'width_ratios':[6,2, 0.8]}, figsize = (10,5))
	#f.tight_layout() 
	g1 = sns.heatmap(decoder_w_np,annot = decoder_w_np.round(3),  cmap = sns.color_palette("coolwarm", 7),
				linewidth=1.5, linecolor = 'black', cbar = True)
	g1.set_title('Decoder Weights (It: {})'.format(iteration))
	plt.savefig(os.path.join(dir_path, 'decoder_wb_it-{}.png'.format(iteration)), bbox_inches='tight')
	model.to(device)

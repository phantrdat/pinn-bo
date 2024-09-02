from matplotlib import pyplot as plt
import matplotlib
import pickle as pkl
import numpy as np
import os
import glob
from objectives import *




D = 50
RNUM = 1000
func_name = 'Cosine_Mixture'
REVERSED = False


directory = f"results/{func_name}_DIM_{D}_ROUNDS_{RNUM}/"

TRIAL = 10
fixed_colors = ['red', 'blue','green', 'black']


colors_map = {'NeuralBO': 'forestgreen', 'NeuralGreedy': 'blue', 'PINN-BO': 'red',  
			 'GPEI': 'grey', 'GPUCB':'purple',}
markers = {'NeuralBO':"*", 'PINN-BO': "v", "GPEI": "^",  'GPUCB': "o",
			'NeuralGreedy': '.'}

def plot_res(directory):

	norm_z = 1
	
	plotting_objs = {'mean':[],'std':[], 'colors': [], 'labels':[], 'markers':[]}
	algs = ["GPEI", "GPUCB",'NeuralGreedy', "NeuralBO", 'PINN-BO']
	for alg in algs: 	
		print(alg)
		res_files = glob.glob(f'{directory}/{alg}/*')

		all_runs = np.empty((0, RNUM+1))
		for pkl_file in res_files[:TRIAL]:
			
			Di  = pkl.load(open(pkl_file,'rb'))	
			optimum_each_run = np.array(Di['optimal_values'])
			
			idx = np.argmin(Di['optimal_values'])
			minimum = optimum_each_run[idx]
			print(pkl_file, minimum)
			
			optimum_each_run = np.array([np.min(optimum_each_run[:i]) for i in range(1, RNUM + 2)])			
			all_runs = np.vstack((all_runs, optimum_each_run))
		runs_std = np.std(all_runs, 0)
		runs_mean = np.mean(all_runs, 0)

		if REVERSED ==True:
			runs_mean = np.array([-v for v in runs_mean])

		plotting_objs['mean'].append(runs_mean)
		plotting_objs['std'].append(runs_std)
		plotting_objs['colors'].append(colors_map[alg])
		plotting_objs['markers'].append(markers[alg])
		plotting_objs['labels'].append(alg)

	for (mean, std, color, label, marker) in zip(plotting_objs['mean'], plotting_objs['std'], plotting_objs['colors'], 
										plotting_objs['labels'], plotting_objs['markers']):
		std =  std[:RNUM+1]
		mean = mean[:RNUM+1]
		start = 0
		end = 1001
		plt.plot(np.arange(start,end), mean[start:end], label=label, color=color, marker=marker, markevery=RNUM//25, markersize=3)
		plt.fill_between(np.arange(start,end), (mean - norm_z*std)[start:end], (mean + norm_z*std)[start:end], alpha=0.1, color=color)
	
	if func_name == 'Heat' or func_name == 'Heat_2' or func_name == 'Heat_3':
		title = 'Temperature Optimization'
	if func_name == "BeamDeflection":
		title = "Beam Displacement Minimization"
	else:
		title = f"{func_name.replace('_',' ')} ({D})"
	plt.title(title, fontsize=14)
	fig = plt.gcf()
	size = fig.get_size_inches()
	desired_aspect_ratio = 8/6
	size[0] = size[1] * desired_aspect_ratio

	fig.set_size_inches(size)
	leg = plt.legend(fontsize=10, bbox_to_anchor=(0.5, -0.25),loc='lower center', ncol=5)
	for legobj in leg.legendHandles:
		legobj.set_linewidth(3.0)

	if func_name == 'Heat' or func_name == 'Heat_2' or func_name == 'Heat_3':
		ylabel = 'Maximum Temperature'
	if func_name == 'BeamDeflection':
		ylabel = "Minimum Displacement"
	else:
		ylabel = "Minimum value observed"
	plt.ylabel(ylabel, fontsize=14)
	plt.xlabel('Number of evaluations', fontsize=14)
	# plt.ylim(bottom=0)
	# plt.xlim(left=0)
	plt.grid()


	fig.tight_layout() 
	if os.path.isdir("figures") ==False:
		os.makedirs("figures")
	fig.savefig(f'figures/{func_name}_dim_{D}_round_{RNUM}.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	fig.savefig(f'figures/{func_name}_dim_{D}_round_{RNUM}.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	print()
	# plt.clf()

if __name__ == '__main__':
	
	plot_res(directory)
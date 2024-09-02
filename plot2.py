from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np


import glob

Ds = [2, 10,20,30, 50]
RNUMS = [100, 1000, 1000, 1000, 1000]
NAMES= ["DropWave",'Styblinski-Tang', 'Rastrigin','Michalewics', 'Cosine_Mixture']
REVERSED = [False,False,False,False,False]




TRIAL = 10
fixed_colors = ['red', 'blue','green', 'black']



colors_map = {'NeuralBO': 'forestgreen', 'NeuralGreedy': 'blue', 'PINN-BO': 'red',  
			 'GPEI': 'grey', 'GPUCB':'purple',}
markers = {'NeuralBO':"*", 'PINN-BO': "v", "GPEI": "^",  'GPUCB': "o",
			'NeuralGreedy': '.'}


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def subplots_centered(nrows, ncols, figsize, nfigs):
	"""
	Modification of matplotlib plt.subplots(),
	useful when some subplots are empty.
	
	It returns a grid where the plots
	in the **last** row are centered.
	
	Inputs
	------
		nrows, ncols, figsize: same as plt.subplots()
		nfigs: real number of figures
	"""
	assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
	
	fig = plt.figure(figsize=figsize)
	axs = []
	
	m = nfigs % ncols
	m = range(1, ncols+1)[-m]  # subdivision of columns
	gs = gridspec.GridSpec(nrows, m*ncols)

	for i in range(0, nfigs):
		row = i // ncols
		col = i % ncols

		if row == nrows-1: # center only last row
			off = int(m * (ncols - nfigs % ncols) / 2)
		else:
			off = 0

		ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
		axs.append(ax)
		
	return fig, axs

def plot_res():
	nrows = 2
	ncols = 3
	figsize = (12,8)
	nfigs = 5

	if nfigs%2 !=0:
		fig, axs = subplots_centered(nrows, ncols, figsize, nfigs)
	else:
		fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
		axs = axs.flatten()
	
	index = 0
	for kf, (func_name,D,RNUM, rv) in enumerate(zip(NAMES,Ds,RNUMS,REVERSED)):
		directory =  f"results/{func_name}_DIM_{D}_ROUNDS_{RNUM}"
		norm_z = 1
		print(func_name)
		plotting_objs = {'mean':[],'std':[], 'colors': [], 'labels':[], 'markers':[]}
		algs = ["GPEI", 'GPUCB', "NeuralGreedy","NeuralBO", 'PINN-BO']

		for alg in algs: 	
			# print(alg)
			res_files = glob.glob(f'{directory}/{alg}/*')

			all_runs = np.empty((0, RNUM+1))
			for pkl_file in res_files[:TRIAL]:
				Di  = pkl.load(open(pkl_file,'rb'))
				optimum_each_run = np.array(Di['optimal_values'])
				optimum_each_run = np.array([np.min(optimum_each_run[:i]) for i in range(1, RNUM + 2)])			
				all_runs = np.vstack((all_runs, optimum_each_run))
			
			runs_std = np.std(all_runs, 0)
			runs_mean = np.mean(all_runs, 0)

			if rv ==True:
				runs_mean = np.array([-v for v in runs_mean])

			plotting_objs['mean'].append(runs_mean)
			plotting_objs['std'].append(runs_std)
			plotting_objs['colors'].append(colors_map[alg])
			plotting_objs['markers'].append(markers[alg])
			if alg.find('_')!=-1:

				plotting_objs['labels'].append(alg[:alg.find('_')])
			else:
				plotting_objs['labels'].append(alg)

		for (mean, std, color, label, marker) in zip(plotting_objs['mean'], plotting_objs['std'], plotting_objs['colors'], 
											plotting_objs['labels'], plotting_objs['markers']):
			std =  std[:RNUM+1]
			mean = mean[:RNUM+1]
			mark_every = RNUM//25
			if rv == True:
				axs[kf].plot(np.arange(0, RNUM+1), 1-mean, label=label, color=color, marker=marker, markevery=mark_every, markersize=3)
				axs[kf].fill_between(np.arange(0, RNUM+1), 1 - mean - norm_z*std, 1- mean + norm_z*std, alpha=0.1, color=color)
			else:
				axs[kf].plot(np.arange(0, RNUM+1), mean[:RNUM+1], label=label, color=color, marker=marker, markevery=mark_every, markersize=3)
				axs[kf].fill_between(np.arange(0, RNUM+1), mean - norm_z*std, mean + norm_z*std, alpha=0.1, color=color)

		axs[kf].set_title(f"{func_name} ({D})", fontsize=12)
		index +=1
	
	

	
	for i, _ in enumerate(range(nfigs)):
		if i%ncols==0:
			axs[i].set_ylabel("Minimum value observed", fontsize=12)
		axs[i].set_xlabel('Number of evaluations', fontsize=12)
		axs[i].grid()

		handles, labels = axs[i].get_legend_handles_labels()
	
	
	leg = fig.legend(handles, labels, fontsize=12, bbox_to_anchor=(0.5, -0.05),loc='lower center', ncol=5)
	for legobj in leg.legendHandles:
		legobj.set_linewidth(1.5)
		

	fig.tight_layout()
	plt.savefig(f'figures/synthetic.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	plt.savefig(f'figures/synthetic.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	plt.clf()

	
if __name__ == '__main__':
	
	plot_res()

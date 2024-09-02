import json
import os

def GP_based(dim, objective_name, acquisition):
    D = {
    "objective_type": "synthetic",    
    "algorithm_type": "GP"+acquisition,
    "function_name": objective_name,
	"acquisition_name": acquisition,
	"dimension" : dim,
	"n_restart": 25,
    "n_raw_samples": 128,
	"n_iter" : 1000,
    "n_init": 50,
	"kernel_name": "Matern",
	"first_run": 1,
	"last_run": 11,
    "normalized_inputs": False,
    "normalized_outputs": False,
    "use_cuda": True
    } 
    dir = f"config/{objective_name}/DIM_{dim}/"
    if os.path.isdir(dir) == False:
        os.makedirs(dir)
    
    outfile = f"config/{objective_name}/DIM_{dim}/{D['algorithm_type']}_{objective_name.lower()}{dim}.json"
    outfile = open(outfile, 'w')
    json.dump(D, outfile, indent = 4)
def NeuralBO(dim, objective_name):
    D = {   
    "objective_type": "synthetic",
	"algorithm_type": "NeuralBO", 
    "function_name": objective_name,
    "dimension" : dim,
	"n_init": 50,
    "n_iter": 1000,

    "L": 2,
    "W": 500,
    "learning_rate": 0.001,
    "epochs": 500,
    "update_cycle": 10,
    "weight_decay": 0.001,
    "use_lr_scheduler":False,
    "activation": "relu",
    "normalized_inputs": True,
    "normalized_outputs": False,

    "use_local_optimizer":True,
    "acqf_optimizer": "Adam",
    "acqf_lr": 0.001,
    "acqf_epochs": 200,
    "n_raw_samples": 512,
    "n_restart": 25,
    

    "exploration_coeff": 0.0,
    "feature_mode": "dynamic",
    "use_matrix_inversion_appoximation": False,
    
    
    "first_run": 1,
    "last_run": 11,
    "use_cuda": True
    }

    dir = f"config/{objective_name}/DIM_{dim}/"
    if os.path.isdir(dir) == False:
        os.makedirs(dir)
    
    outfile = f"config/{objective_name}/DIM_{dim}/{D['algorithm_type']}_{objective_name.lower()}{dim}.json"
    outfile = open(outfile, 'w')
    json.dump(D, outfile, indent = 4)
def NeuralGreedy(dim, objective_name):
    D = {   
    "objective_type": "synthetic",
	"algorithm_type": "NeuralGreedy", 
    "function_name": objective_name,
    "dimension" : dim,

	"n_init": 50,
    "n_iter": 20,
    
    "L": 2,
    "W": 500,
    "learning_rate": 0.001,
    "epochs": 500,
    "update_cycle": 10,
    "scale_parameter": 0.1,
    "use_lr_scheduler":False,
    "activation": "relu",
    "normalized_inputs": True,
    "normalized_outputs": False,

    "use_local_optimizer":True,
    "acqf_optimizer": "Adam",
    "acqf_lr": 0.001,
    "acqf_epochs": 512,
    "n_raw_samples": 1000,
    "n_restart": 25,
    
    "first_run": 1,
    "last_run": 11,
    "use_cuda": True
    }
    dir = f"config/{objective_name}/DIM_{dim}/"
    if os.path.isdir(dir) == False:
        os.makedirs(dir)
    
    outfile = f"config/{objective_name}/DIM_{dim}/{D['algorithm_type']}_{objective_name.lower()}{dim}.json"
    outfile = open(outfile, 'w')
    json.dump(D, outfile, indent = 4)
def PINN_NeuralBO(dim, objective_name):
    D = {   
    "objective_type": "synthetic",
	"algorithm_type": "PINN_NeuralBO", 
    "function_name": objective_name,
    "dimension" : dim,
	"n_residuals": 40000,
    "n_init": 50,
    "n_iter": 1000,

    "L": 4,
    "W": 50,
    "learning_rate": 0.01,
    "epochs": 200,
    "update_cycle": 10,
    "weight_decay": 0.001,
    "use_lr_scheduler":False,
    "activation": "tanh",
    "normalized_inputs": True,
    "normalized_outputs": False,

    "use_local_optimizer":True,
    "acqf_optimizer": "Adam",
    "acqf_lr": 0.001,
    "acqf_epochs": 512,
    "n_raw_samples": 50000,
    "n_restart": 100,
    
    "first_run": 1,
    "last_run": 11,
    "use_cuda": True
    }
    dir = f"config/{objective_name}/DIM_{dim}/"
    if os.path.isdir(dir) == False:
        os.makedirs(dir)
    
    outfile = f"config/{objective_name}/DIM_{dim}/{D['algorithm_type']}_{objective_name.lower()}{dim}.json"
    outfile = open(outfile, 'w')
    json.dump(D, outfile, indent = 4)




# def DNGO(dim, objective_name):
#     D = {   
#     "function_type": "syn",
# 	"algorithm_type": "DNGO", 
#     "function_name": objective_name,
# 	"dimension" : dim,
# 	"num_epochs": 100,
# 	"learning_rate": 0.01,
# 	"hidden_size": 10,
# 	"n_iter" : 2000,
# 	"n_restart": 20, 
# 	"first_run": 1,
# 	"last_run": 11
#     }
#     dir = f"config/{objective_name}/DIM_{dim}/"
#     if os.path.isdir(dir) == False:
#         os.makedirs(dir)
    
#     outfile = f"config/{objective_name}/DIM_{dim}/{D['algorithm_type']}_{objective_name.lower()}{dim}.json"
#     outfile = open(outfile, 'w')
#     json.dump(D, outfile, indent = 4)

# # def RF(dim, objective_name):
#     D = {   
#     "function_type": "syn",
# 	"algorithm_type": "RF", 
#     "function_name": objective_name,
# 	"dimension" : dim,
# 	"n_restart": 20,
# 	"n_iter" : 2000,
# 	"first_run": 1,
# 	"last_run": 11
#     }
#     dir = f"config/{objective_name}/DIM_{dim}/"
#     if os.path.isdir(dir) == False:
#         os.makedirs(dir)
    
#     outfile = f"config/{objective_name}/DIM_{dim}/{D['algorithm_type']}_{objective_name.lower()}{dim}.json"
#     outfile = open(outfile, 'w')
#     json.dump(D, outfile, indent = 4)
def gen_script(dim, objective_name):
    import glob
    from pathlib import Path

    root = 'config/'
    func = f'{objective_name}/DIM_{dim}'
    files  = glob.glob(os.path.join(root,func,"*"))
    script_dir = f'run_scripts/{func}'
    if os.path.isdir(script_dir)==False:
        os.makedirs(script_dir)
    for f in files: 
        script_name = Path(f).stem + '.sh'
        if os.path.isdir(f"logs/{func}") == False:
            os.makedirs(f"logs/{func}")
        
        # if 'neural' in f.lower():
        if 'rf' in f.lower() or 'dngo' in f.lower():
            main_cmd = f'/home/trongp/.conda/envs/pytorch_BO/bin/python main_gpu.py -cfg {f}'
            cmd0 = f'#SBATCH --partition=cpu --ntasks=1 --cpus-per-task=1 --output=logs/{func}/slurm_%j.out\n'
        

        else:

            cmd0 = f'#SBATCH --partition=gpu --gpus=1 --cpus-per-gpu=1 --output=logs/{func}/slurm_%j.out\n'
            if "gp" in f.lower():
                main_cmd = f'/home/trongp/.conda/envs/torch_bo/bin/python main_gpu_new.py -cfg {f}'
            else:
                main_cmd = f'/home/trongp/.conda/envs/pytorch_BO/bin/python main_gpu_new.py -cfg {f}'
            main_cmd += ' -gpu_id 0\n'
        # else:
        #     cmd0 = '#SBATCH --partition=cpu --ntasks=1 --cpus-per-task=1 --output=logs/slurm_%j.out\n'
            
        cmd_list = ['#!/bin/bash\n',cmd0,
                'module load Anaconda3\n', 'source activate pytorch_BO\n', main_cmd]
        print(os.path.join(script_dir, script_name.lower()))
        f = open(os.path.join(script_dir, script_name.lower()),'w')
        f.writelines(cmd_list)
if __name__=='__main__':
    dim = [1]
    fnames= ['BeamDeflection']
    for (d,fname) in zip(dim, fnames):
        GP_based(d, fname, 'EI')
        GP_based(d, fname, 'UCB')
        NeuralBO(d, fname)
        NeuralGreedy(d, fname)
        PINN_NeuralBO(d, fname)
        gen_script(d, fname)

             


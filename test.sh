#!/bin/bash
#SBATCH --job-name=start_train
#SBATCH --partition=gpu #set to GPU for GPU usage
#SBATCH --nodes=1             # number of nodes
#SBATCH --mem=60GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/kmanhu2s/Test/surface_normal_uncertainty-main/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/kmanhu2s/Test/surface_normal_uncertainty-main/job_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda

# activate environment and create marker file
source ~/Test/venv/bin/activate && touch venv_activated.marker  #Uncomment to run in python VENV

# add the correct path to LD_LIBRARY_PATH without spaces around '='
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib/


#Uncomment to run in python VENV

# check if the virtual environment is activated
if [ -e venv_activated.marker ]; then
    echo "Virtual environment is activated."
    # locate to your root directory
	cd /home/kmanhu2s/Test/surface_normal_uncertainty-main
	# run the script
	python train.py --n_epochs 7 --exp_name epoch7

else
    echo "Virtual environment is not activated."
fi

# Uncomment to run in python VENV


# # Uncomment to run in conda

# # add the correct path to LD_LIBRARY_PATH without spaces around '='
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib/
# # activate environment and create marker file
# source activate /home/kmanhu2s/anaconda3/envs/masters
# # locate to your root directory
# cd /home/kmanhu2s/Test/surface_normal_uncertainty-main
# # run the script
# python train.py --n_epochs 15 --exp_name epoch15	
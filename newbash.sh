#!/bin/bash
#SBATCH --job-name=log50_4BS_NLL_vMF_loss
#SBATCH --partition=gpu4      #set to GPU for GPU usage
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1            # number of nodes
#SBATCH --mem=120GB               # memory per node in MB (different units with$

#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /work/kmanhu2s/surface_normal_uncertainty-main/jobs/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /work/kmanhu2s/surface_normal_uncertainty-main/jobs/job_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda



source /home/kmanhu2s/anaconda3/etc/profile.d/conda.sh

conda activate python38

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib/

echo $CONDA_DEFAULT_ENV

cd /home/kmanhu2s/Test/surface_normal_uncertainty-main

python3 trainlog.py --n_epochs 50 --exp_name log50_4BS_NLL_vMF_loss --batch_size 4 --loss_fn NLL_vMF --exp_dir /work/kmanhu2s/surface_normal_uncertainty-main/trials

#!/bin/bash
#SBATCH --job-name="hislam2_run_demo"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/storage/user/ayu/slurm/logs/hislam-%j.out
#SBATCH --exclude=node1,node7

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hislam_slurm
python setup.py install
# Start TensorBoard in the background
# To view live: ssh -L 6006:$(hostname):6006 <your-username>@<cluster-address> and open http://localhost:6006
# nohup tensorboard --logdir outputs/scene0000_00/tensorboard --port 6006 &

python demo.py \
--imagedir data/Replica/room0/colors \
--calib calib/replica.txt \
--config config/replica_config.yaml \
--output outputs/semantic/room0_10
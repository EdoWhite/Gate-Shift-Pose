#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --mem=80G
#SBATCH --partition=gpu-pre
#SBATCH --account=pro-est
#SBATCH --output=./out/example-%j.out
#SBATCH --error=./out/example-%j.err
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=5-00:00:00

module load python/3.8.15-aocc-3.2.0-linux-ubuntu22.04-zen2
module load cuda/11.8.0-gcc-12.1.0-6ihpcg2
source compvis/bin/activate

cd WinterSport/GSF-Pose
python main.py --arch resnet50 --batch-size 4 --num_segments 16 --lr 0.01 --epochs 120 --dropout 0.5 --weight-decay 5e-4 --warmup 10  --num_channels 3 --gsf --gsf_ch_ratio 100 --dataset FRFS --dataset_path /data/users/edbianchi/FRFS_BAK/dataset --experiment_path /data/users/edbianchi/WinterSport --experiment_name FRFS-B4-S16-001-120-vanilla -j 8
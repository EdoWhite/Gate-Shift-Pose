#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --mem=80G
#SBATCH --partition=gpu-pre
#SBATCH --account=pro-est
#SBATCH --output=./out/example-%j.out
#SBATCH --error=./out/example-%j.err
#SBATCH --gres=gpu:a100-sxm4-80gb:1

module load python/3.8.15-aocc-3.2.0-linux-ubuntu22.04-zen2
module load cuda/11.8.0-gcc-12.1.0-6ihpcg2
source compvis/bin/activate

cd WinterSport/GSF
python main.py --arch resnet50 --num_segments 4 --dropout 0.5 --warmup 10  --gsf --gsf_ch_ratio 100 --dataset FRFS --dataset_path /data/users/edbianchi/FRFS/dataset --epochs 60 --batch-size 16 --lr 0.001 --experiment_path /data/users/edbianchi/WinterSport --experiment_name testFRFS-B16-S16-0001-R50-W-POSEs -j 0
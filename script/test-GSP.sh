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
python test_models.py --arch resnet50 --num_channels 4 --num_segments 32 --early_fusion_poses --num_clips 2 --test_crops 10 --gsf --gsf_ch_ratio 100 --dataset FRFS --dataset_path /data/users/edbianchi/FRFS_BAK/dataset --weights /data/users/edbianchi/WinterSport/experiments/FRFS/resnet50/Nov15_09-15-20_FRFS-B4-S32-001-200-POSE/FRFS_resnet50_segment32_best.pth.tar -j 8
#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --account=u430201701    # 你实际有权限的项目账号
#SBATCH --gres=gpu:2
#SBATCH --time=512:00:00                  # 最长运行时间
#SBATCH --mem=256G                        # 分配内存
#SBATCH --nodelist=hpcgpu06              # 指定节点
eval "$(/home/mail/2023t3/u430201701/anaconda3/bin/conda shell.bash hook)"
conda activate vmfgai
# ✅ 加载高版本 GCC，避免 cpu_adam 编译失败
module load gcc12
# 可选：清缓存（防止上次失败的编译缓存影响）
rm -rf ~/.cache/torch_extensions/
export LD_LIBRARY_PATH=/home/mail/tieyongzeng/anaconda3/lib:/opt/clustertech/chess/ng/bin
# ✅ 屏蔽掉 GPU 1
#export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
cd "/home/mail/2023t3/u430201701/hxproject/Copy_MegablocksGavaMOE-vmf/"
deepspeed train.py --deepspeed ds_config.json
#python train.py 
nvidia-smi
sleep 10
hostname
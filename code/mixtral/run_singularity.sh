#!/bin/bash
#SBATCH --job-name=mixtral_moe
#SBATCH --output=output_mixtral_moe.txt
#SBATCH --error=error_mixtral_moe.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=48:0:0
#SBATCH --partition=gpu
#SBATCH --gpus=1

singularity exec \
    --nv \
    --bind ./:/src/ \
    --bind ../data/:/src/data \
    ./singularity/outbrain_ctr.sif \
    python3 \
    main.py 

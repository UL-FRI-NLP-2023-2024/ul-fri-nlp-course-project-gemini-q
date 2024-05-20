#!/bin/bash
#SBATCH --job-name=mistralai/Mixtral-8x7B-v0.1_nlp
#SBATCH --output=output_mixtral_moe.txt
#SBATCH --error=error_mixtral_moe.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=2:0:0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodelist=gwn[01-07]

singularity exec \
    --nv \
    --bind ./:/src/ \
    --bind /d/hpc/projects/FRI/spagnolog/hf-cache/:/hf-cache \
    ./singularity/nvidia_container.sif \
    python3 \
    /src/ft.py


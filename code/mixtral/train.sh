#!/bin/bash
#SBATCH --job-name=mistralai/Mixtral-8x7B-v0.1_medical_SLO_finetune_nlp_la_baugette
#SBATCH --output=output_mixtral_moe.txt
#SBATCH --error=error_mixtral_moe.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=72:00:0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodelist=gwn[01-07]

singularity exec \
    --nv \
    --bind ./:/src/ \
    --bind /d/hpc/projects/FRI/spagnolog/hf-cache/:/hf-cache \
    --bind /d/hpc/projects/FRI/spagnolog/models:/models \
    ./singularity/nvidia_container.sif \
    python3 \
    /src/ft.py

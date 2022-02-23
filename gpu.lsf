#!/bin/bash
#BSUB -q gpu
#BSUB -W 24:00
#BSUB -J NaMgMNO
#BSUB -o %J.stdout
#BSUB -e %J.stderr
#BSUB -n 4
#BSUB -gpu "num=1:mode=shared:mps=no:j_exclusive=yes"
#BSUB -R "span[ptile=4]"

# add modulefiles & activate virtual environment
conda activate tf-gpu
python gpu.py

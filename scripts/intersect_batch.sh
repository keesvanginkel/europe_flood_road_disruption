#!/bin/bash
#
#SBATCH --job-name=intersect_EU
#SBATCH --partition=ivm
#SBATCH --ntasks=1
#SBATCH --time=02-00
#SBATCH --cpus-per-task=10
#SBATCH --output=out_inter
#SBATCH --error=err_inter
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elco.koks@vu.nl

source activate py38
python /scistor/ivm/eks510/projects/europe_flood_road_disruption/scripts/intersect_floods_edgeid.py
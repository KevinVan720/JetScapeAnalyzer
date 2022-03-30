#!/bin/bash

source /usr/local/init/profile.sh
module load /heppy/modules/heppy/1.0
python3 /global/cscratch1/sd/wf39/Analysis/JetScapeAnalyzer/read_folder.py $1 $2 $3


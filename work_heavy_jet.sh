#!/bin/bash

source /usr/local/init/profile.sh
module load /heppy/modules/heppy/1.0
python read_folder_heavy_jet.py /home/input/ /home/output/ $1 $2


import numpy as np
import os, sys
from os import path
import pathlib
import h5py
import fnmatch
import multiprocessing
from joblib import Parallel, delayed
import re
from itertools import groupby
import time

homeDir="/tier2/home/groups/jetscape/gy8046/"

baseDir=str(pathlib.Path(__file__).parent.absolute())+"/"

dir_name=baseDir.replace("/Analysis","")+"OutputFiles/"

batchIndex=range(0,100,10)

queueType="primary"

for batch in batchIndex:

    jobFolder=baseDir+"RUN_"+str(batch)+"/"
    os.makedirs(jobFolder, exist_ok=True)
    os.system("cp "+homeDir+"Analysis/JetScapeAnalyzer/*.py "+ jobFolder) 
    os.system("cp "+homeDir+"Analysis/JetScapeAnalyzer/work.sh "+ jobFolder)
    
    subFileName = jobFolder+"sub_job" + ".sh"
    subFile = open(subFileName, "w")
    subFile.writelines("#!/usr/bin/env bash\n")
    #subFile.writelines("#PBS -S /bin/bash\n")
    subFile.writelines("#SBATCH -q "+queueType+"\n")
    #subFile.writelines("#PBS -l cpu_type=Intel\n")
    #subFile.writelines("#PBS -l cpu_model=6148\n")
    subFile.writelines("#SBATCH --mem=4GB\n")
    subFile.writelines("#SBATCH -n 10\n")

    subFile.writelines("#SBATCH --job-name=folder_" + str(batch) + "\n")
    subFile.writelines("#SBATCH -e folder_" + str(batch) + ".err\n")
    subFile.writelines("#SBATCH -o folder_" + str(batch) + ".log\n")
    subFile.writelines("#SBATCH --time 100:00:00\n")
    # subFile.writelines("#PBS -A cqn-654-ad\n")

    subFile.writelines(
        "singularity run -B "
        + dir_name
        + ":/home/input/,"
        + jobFolder
        + ":/home/output/" 
        +" "+homeDir+"Analysis/jetscape_analysis_latest.sif"
        +" bash /home/output/work.sh "
        + str(batch)
        + " 1> "+jobFolder+"RUN_"
        + str(batch)
        + ".log 2> "+jobFolder+"RUN_"
        + str(batch)
        + ".err\n"
    )

    subFile.close()

    # Now, submit file
    os.system("cd " + jobFolder + " && sbatch " + subFileName + " > " + subFileName + ".out")

    # ...and wait a few seconds, so as not to overwhelm the scheduler
    time.sleep(0.5)


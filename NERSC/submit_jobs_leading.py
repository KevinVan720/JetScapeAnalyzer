import numpy as np
import os
import sys
from os import path
import pathlib
import h5py
import fnmatch
import multiprocessing
from joblib import Parallel, delayed
import re
from itertools import groupby
import time

homeDir = "/global/cscratch1/sd/wf39/"

baseDir = str(pathlib.Path(__file__).parent.absolute())+"/"

dir_name = baseDir.replace("/Analysis", "/Simulation")+"OutputFiles/"

batchIndex = range(0, 100, 10)

queueType = "regular"


def run_analysis(inputDir, OutputDir, batch):
    os.makedirs(OutputDir, exist_ok=True)
    #os.system("cp "+homeDir+"Analysis/JetScapeAnalyzer/*.py " + OutputDir)
    
    os.system("cp "+homeDir+"Analysis/JetScapeAnalyzer/NERSC/work*.sh " + OutputDir)

    subFileName = OutputDir+"sub_job" + ".sh"
    subFile = open(subFileName, "w")
    subFile.writelines("#!/usr/bin/env bash\n")
    subFile.writelines("#SBATCH -q "+queueType+"\n")
    subFile.writelines("#SBATCH --mem=4GB\n")


    subFile.writelines("#SBATCH --constraint=knl\n")
    subFile.writelines("#SBATCH -N 1\n")
    subFile.writelines("#SBATCH --license cscratch1")

    subFile.writelines("#SBATCH --job-name=folder_" + str(batch) + "\n")
    subFile.writelines("#SBATCH -e folder_" + str(batch) + ".err\n")
    subFile.writelines("#SBATCH -o folder_" + str(batch) + ".log\n")
    subFile.writelines("#SBATCH --time 36:00:00\n")

    subFile.writelines("#SBATCH --image=docker:wenkaifan/jetscape_analyzer:latest\n")

    subFile.writelines(
        "srun -n 10 shifter "+OutputDir+"/work_leading.sh "
        + inputDir
        + " "
        + OutputDir
        + " "
        + str(batch)
        + " 1> "+OutputDir+"RUN_"
        + str(batch)
        + ".log 2> "+OutputDir+"RUN_"
        + str(batch)
        + ".err\n"
    )

    subFile.close()

    # Now, submit file
    os.system("cd " + OutputDir + " && sbatch " +
              subFileName + " > " + subFileName + ".out")

    # ...and wait a few seconds, so as not to overwhelm the scheduler
    time.sleep(0.5)


for batch in batchIndex:

    jobFolder = baseDir+"RUN_"+str(batch)+"/"
    os.makedirs(jobFolder, exist_ok=True)

    run_analysis(dir_name, jobFolder, batch)
    #run_analysis(dir_name+"/D/", jobFolder+"/D/", batch)
    #run_analysis(dir_name, jobFolder+"/DB/", "DB", batch)

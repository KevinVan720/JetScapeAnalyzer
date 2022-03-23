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

homeDir = "/tier2/home/groups/jetscape/gy8046/"

baseDir = str(pathlib.Path(__file__).parent.absolute())+"/"

dir_name = baseDir.replace("/Analysis", "")+"OutputFiles/"

batchIndex = range(0, 100, 10)

queueType = "primary"


def run_analysis(inputDir, OutputDir, suffix, batch):
    os.makedirs(OutputDir, exist_ok=True)
    os.system("cp "+homeDir+"Analysis/JetScapeAnalyzer/*.py " + OutputDir)
    os.system("cp "+homeDir+"Analysis/JetScapeAnalyzer/work*.sh " + OutputDir)

    subFileName = OutputDir+"sub_job" + ".sh"
    subFile = open(subFileName, "w")
    subFile.writelines("#!/usr/bin/env bash\n")
    subFile.writelines("#SBATCH -q "+queueType+"\n")
    subFile.writelines("#SBATCH --mem=4GB\n")
    subFile.writelines("#SBATCH -n 10\n")

    subFile.writelines("#SBATCH --job-name=folder_" + str(batch) + "\n")
    subFile.writelines("#SBATCH -e folder_" + str(batch) + ".err\n")
    subFile.writelines("#SBATCH -o folder_" + str(batch) + ".log\n")
    subFile.writelines("#SBATCH --time 100:00:00\n")

    subFile.writelines(
        "singularity run -B "
        + inputDir
        + ":/home/input/,"
        + OutputDir
        + ":/home/output/"
        + " "+homeDir+"Analysis/jetscape_analysis_latest.sif"
        + " bash /home/output/work_leading.sh "
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

    run_analysis(dir_name, jobFolder+"/light/", "light", batch)
    run_analysis(dir_name, jobFolder+"/D/",  "D", batch)
    #run_analysis(dir_name, jobFolder+"/DB/", "DB", batch)

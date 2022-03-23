import numpy as np
import os, sys
from os import path
import h5py
import fnmatch
import multiprocessing
from joblib import Parallel, delayed
import re
import subprocess
from itertools import groupby
import math

def processpThatBin(files):

    files.sort()
    for fileID in range(0, len(files)):
        fileName = files[fileID]
        print("Reading: ", fileName)

        csFileName = [file for file in XsectionFiles if keyf(file) == keyf(fileName)][0]
        weight = np.loadtxt(dir_name + csFileName)[0]
        subprocess.run(["./2Dhydro-jet-analysis", dir_name + fileName, str(weight)])
        

if __name__ == "__main__":

    folderID = sys.argv[1]
    dir_name = sys.argv[2]
    print(folderID, dir_name)

    #baseDir="/wsu/home/gy/gy80/DJet_cen0-10-Q0/"

    num_cores = multiprocessing.cpu_count()

    allFiles = os.listdir(dir_name)
    # files stores all the event output hadrons
    files = [file for file in allFiles if fnmatch.fnmatch(file, "*Hadrons*.dat")]
    files.sort()

    # XsectionFiles stores all the event sigmagen
    XsectionFiles = [
        file for file in allFiles if fnmatch.fnmatch(file, "*Xsection*.dat")
    ]
    XsectionFiles.sort()

    # keyf is used to group events with the same pThatbin
    keyf = lambda text: (re.findall("\d{4}_\d{4}", text) + [text])[0]
    pThatBinFiles = [list(items) for gr, items in groupby(sorted(files), key=keyf)]

    # Use this many cores to process pThatBin files in parallel
    print("Analyzing folder", dir_name)
    print("Cores used: ", num_cores)

    # this is the main program, process all the pThatBin files in parallel
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(processpThatBin)(i) for i in pThatBinFiles
    )


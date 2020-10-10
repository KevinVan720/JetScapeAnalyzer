import numpy as np
import os
import sys
from os import path
import h5py
import fnmatch
import multiprocessing
from joblib import Parallel, delayed
import re
from itertools import groupby
import math
import random
from analysis import pTYieldAnalysis
from analysis import FlowAnalysis
from analysis import JetScapeReader
from analysis import InclusiveJetpTYieldAnalysis


def doAnalysisOnBatch(batchIndexStart, batchIndexEnd):
    pThatPair = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
                 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2510]
    pThat_Min = pThatPair[:-1]
    pThat_Max = pThatPair[1:]

    allAnalysis = [
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100, 200, 300, 500],
                        ids=[211, -211, 213, -213, 321, -321, 2212, -2212, 3222, -3222, 3112, -3112, 3312, -3312, 3334, -3334], rapidityCut=[-1, 1], outputFileName=outputDir+"/ch_yield_"+"%06d" % random.randint(0, 999999)+".txt"),
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
                        ids=[421, -421], rapidityCut=[-1, 1], outputFileName=outputDir+"/D0_yield_"+"%06d" % random.randint(0, 999999)+".txt"),
        InclusiveJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
                                    rapidityCut=[-1, 1],
                                    jetRadius=0.3, jetpTMin=1, jetRapidityCut=[-2, 2],
                                    outputFileName=outputDir+"/inclusiveJet_yield_"+"%06d" % random.randint(0, 999999)+".txt"),
        FlowAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 15, 20, 40],
                     ids=[411, -411, 421, -421, 413, -413, 423, -423], rapidityCut=[-1, 1], outputFileName=outputDir+"/D0_v2_"+"%06d" % random.randint(0, 999999)+".txt"),
    ]

    assert(batchIndexStart >= 0 and batchIndexStart < batchIndexEnd)

    for batchIndex in range(batchIndexStart, batchIndexEnd):
        for pThatIndex in range(len(pThat_Min)):
            pThatString = "%04d" % pThat_Min[pThatIndex] + \
                "_"+"%04d" % pThat_Max[pThatIndex]
            batchString = "%03d" % (batchIndex*10) + \
                "k_to_"+"%03d" % ((batchIndex+1)*10)
            #print(pThatString, batchString)
            files = [
                file for file in allFiles if batchString in file and pThatString in file]
            for fileName in files:
                print("Reading: ", fileName)
                # reading a certain file, add one event number as we encounter line starting with #
                reader = JetScapeReader(inputDir+fileName)
                for hadrons in reader.readAllEvents():
                    for analysis in allAnalysis:

                        analysis.setStatus(pThatIndex, reader)
                        CSFile = [file for file in os.listdir(inputDir) if fnmatch.fnmatch(
                            file, "Xsection_*"+pThatString+"*.dat")]
                        if len(CSFile) > 0:
                            sigma = np.loadtxt(inputDir+CSFile[0])[0]
                            analysis.setCrossSection(pThatIndex, sigma)

                        analysis.analyzeEvent(hadrons)

    for analysis in allAnalysis:
        analysis.outputResult()


if __name__ == "__main__":

    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    batchIndex = int(sys.argv[3])

    num_cores = multiprocessing.cpu_count()

    allFiles = os.listdir(inputDir)
    # files stores all the event output hadrons
    allFiles = [file for file in allFiles if fnmatch.fnmatch(
        file, "*Hadrons*.dat")]
    allFiles.sort()

    # Use this many cores to process pThatBin files in parallel
    print("Analyzing folder", inputDir)
    print("Cores used: ", num_cores)

    # prepareHydro(files[0])

    # this is the main program, process all the pThatBin files in parallel
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(doAnalysisOnBatch)(i, i+1) for i in range(batchIndex+0, batchIndex+9))

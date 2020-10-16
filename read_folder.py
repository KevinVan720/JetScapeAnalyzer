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
from analysis import *
import subprocess


def doAnalysisOnBatch(batchIndexStart, batchIndexEnd):
    pThatPair = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
                 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2510]
    pThat_Min = pThatPair[:-1]
    pThat_Max = pThatPair[1:]

    allAnalysis = [
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100, 200, 300, 500],
                        ids=[211, -211, 213, -213, 321, -321, 2212, -2212, 3222, -3222, 3112, -3112, 3312, -3312, 3334, -3334], rapidityCut=[-1, 1], outputFileName=outputDir+"ch_yield"),
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100, 200, 300, 500],
                        ids=[211, -211, 213, -213, 321, -321, 2212, -2212, 3222, -3222, 3112, -3112, 3312, -3312, 3334, -3334], rapidityCut=[-0.5, 0.5], outputFileName=outputDir+"ch_yield"),
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
                        ids=[421, -421], rapidityCut=[-1, 1], outputFileName=outputDir+"D0_yield"),
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
                        ids=[421, -421], rapidityCut=[-0.5, 0.5], outputFileName=outputDir+"D0_yield"),
        FlowAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 15, 20, 40],
                     ids=[411, -411, 421, -421, 413, -413, 423, -423], rapidityCut=[-1, 1], outputFileName=outputDir+"D0_v2"),
        InclusiveJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100, 250, 300, 350],
                                    rapidityCut=[-1, 1],
                                   jetRadius=0.3, jetpTMin=1, jetRapidityCut=[-2, 2],
                                   outputFileName=outputDir+"inclusiveJet_yield"),
        HeavyJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360],
                                jetRadius=0.2, jetEtaCut=[-2.1, 2.1], drCut=0.3, heavypTMin=5, ids=[411, -411, 421, -421, 413, -413, 423, -423], outputFileName=outputDir+"DJet_yield"),
        HeavyRadialProfileAnalysis(pThatBins=pThatPair, rBins=[0, 0.05, 0.1, 0.3, 0.5], jetRadius=0.3, jetpTMin=60, jetEtaCut=[-1.6, 1.6], heavypTMin=20, heavyEtaCut=[-2, 2], ids=[-421, 421],
                                   outputFileName=outputDir+"D0_jet_radialprofile"),
    ]

    assert(batchIndexStart >= 0 and batchIndexStart < batchIndexEnd)

    for batchIndex in range(batchIndexStart, batchIndexEnd):
        for pThatIndex in range(len(pThat_Min)):
            pThatString = str(pThat_Min[pThatIndex]).zfill(4) + \
                "_" + str(pThat_Max[pThatIndex]).zfill(4)
            batchString = str(batchIndex*10).zfill(3) + \
                "k_to_"+str((batchIndex+1)*10).zfill(3)

            files = [
                file for file in allFiles if batchString in file and pThatString in file]
            for fileName in files:
                
                # reading a certain file, add one event number as we encounter line starting with #
                reader = JetScapeReader(inputDir+fileName)
                for hadrons in reader.readAllEvents():
                    for analysis in allAnalysis:

                        # no hydro information? Can not do flow analysis
                        if reader.currentHydroInfo==[]:
                            assert(analysis is not FlowAnalysis)

                        if reader.currentCrossSection > 0:
                            analysis.setStatus(pThatIndex, reader)
                        else:
                            sigma = getCrossSectionFromFile(
                                inputDir, "Xsection_*"+pThatString+"*.dat")
                            analysis.setCrossSection(pThatIndex, sigma)

                        analysis.analyzeEvent(hadrons)
            
            #temp output file to see anything wrong
            for analysis in allAnalysis:
                analysis.outputResult()
                

    for analysis in allAnalysis:
        analysis.outputResult()


if __name__ == "__main__":

    inputDir = sys.argv[1]+"/"
    outputDir = sys.argv[2]+"/"
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

    # this is the main program, process all the pThatBin files in parallel
    batchExtend = 1
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(doAnalysisOnBatch)(i, i+batchExtend) for i in range(batchIndex+0, batchIndex+9, batchExtend))


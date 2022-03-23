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

    chpTBins=[0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 3.2, 4, 4.8, 5.6, 6.4, 7.2, 9.6, 12, 14.4, 19.2, 24, 28.8, 35.2, 41.6, 48, 60.8, 73.6, 86.4, 103.6, 120.8, 140, 165, 250, 400]

    DmesonIds={411, -411, 421, -421, 413, -413, 423, -423}


    allAnalysis = [        
        HeavyJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360],
                               jetRadius=0.3, jetEtaCut=[-2.1, 2.1], drCut=0.3, heavypTCut=[5, 30000], ids=DmesonIds, outputFileName=outputDir+"DJet_yield"),
        HeavyRadialProfileAnalysis(pThatBins=pThatPair, rBins=[0, 0.05, 0.1, 0.3, 0.5], jetRadius=0.3, jetpTMin=60, jetEtaCut=[-1.6, 1.6], heavypTCut=[20, 30000], heavyEtaCut=[-2, 2], ids=[-421, 421],
                                   outputFileName=outputDir+"D0_jet_radialprofile"),
        HeavyRadialProfileAnalysis(pThatBins=pThatPair, rBins=[0, 0.05, 0.1, 0.3, 0.5], jetRadius=0.3, jetpTMin=60, jetEtaCut=[-1.6, 1.6], heavypTCut=[4, 20], heavyEtaCut=[-2, 2], ids=[-421, 421],
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
            headerFiles= [
                headerFile for headerFile in allHeaderFiles if batchString in headerFile and pThatString in headerFile]
            for fileName in files:

                # reading a certain file, add one event number as we encounter line starting with #
                reader = JetScapeReader(inputDir+"/"+fileName, headerName=inputDir+"/"+headerFiles[0])
                for hadrons in reader.readAllEvents():
                    for analysis in allAnalysis:

                        # no hydro information? Can not do flow analysis
                        if reader.currentHydroInfo == []:
                            assert(analysis is not FlowAnalysis)
                        #print(reader.headerName,reader.currentCrossSection)
                        analysis.setStatus(pThatIndex,reader)
                        '''if reader.currentCrossSection > 0:
                            analysis.setStatus(pThatIndex, reader)
                        else:
                            sigma = getCrossSectionFromFile(
                                inputDir, "Xsection_*"+pThatString+"*.dat")
                            analysis.setCrossSection(pThatIndex, sigma)
                        '''
                        analysis.analyzeEvent(hadrons)
                
                #os.remove(inputDir+"/"+sys.argv[4]+"/"+fileName)

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

    allHeaderFiles = os.listdir(inputDir)
    # files stores all the event output hadrons
    allHeaderFiles = [file for file in allHeaderFiles if fnmatch.fnmatch(
        file, "*Header*.dat")]
    allHeaderFiles.sort()

    # Use this many cores to process pThatBin files in parallel
    print("Analyzing folder", inputDir)
    print("Cores used: ", num_cores)

    # this is the main program, process all the pThatBin files in parallel
    batchExtend = 1
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(doAnalysisOnBatch)(i, i+batchExtend) for i in range(batchIndex+0, batchIndex+9, batchExtend))

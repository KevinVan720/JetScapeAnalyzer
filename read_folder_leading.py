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
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=chpTBins,
                        ids=chargeHadronIds, etaCut=[-1, 1], outputFileName=outputDir+"ch_yield"),
        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=chpTBins,
        #                ids=chargeHadronId, etaCut=[-0.5, 0.5], outputFileName=outputDir+"ch_yield"),
        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=chpTBins,
        #                ids=chargeHadronId, etaCut=[-0.1, 0.1], outputFileName=outputDir+"ch_yield"),

        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
        #                ids={421, -421}, rapidityCut=[-1, 1], outputFileName=outputDir+"D0_yield"),
        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
        #                ids={421, -421}, rapidityCut=[-0.5, 0.5], outputFileName=outputDir+"D0_yield"),
        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
        #                ids=[421, -421], rapidityCut=[-0.1, 0.1], outputFileName=outputDir+"D0_yield"),
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100, 150, 200, 250, 300, 350, 400, 500],
                        ids={421, -421}, rapidityCut=[-1, 1], outputFileName=outputDir+"D0_yield"),

        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
        #                ids={411, -411, 421, -421, 413, -413, 423, -423}, rapidityCut=[-1, 1], outputFileName=outputDir+"D_yield"),
        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
        #                ids={411, -411, 421, -421, 413, -413, 423, -423}, rapidityCut=[-0.5, 0.5], outputFileName=outputDir+"D_yield"),
        #pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100],
        #                ids=[421, -421], rapidityCut=[-0.1, 0.1], outputFileName=outputDir+"D0_yield"),
        pTYieldAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 30, 40, 60, 100, 150, 200, 250, 300, 350, 400, 500],
                        ids=DmesonIds, rapidityCut=[-1, 1], outputFileName=outputDir+"D_yield"),

        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[10,15],pTCut2=[8,12],ids1={421,-421},ids2={421,-421},useAnti=True, outputFileName=outputDir+"D0D0bar_correlation"),
        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[4,6],pTCut2=[2,5],ids1={421,-421},ids2={421,-421},useAnti=True, outputFileName=outputDir+"D0D0bar_correlation"),


        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[10,15],pTCut2=[8,12],ids1=DmesonIds,ids2=DmesonIds,useAnti=True, outputFileName=outputDir+"DDbar_correlation"),
        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[4,6],pTCut2=[2,5],ids1=DmesonIds,ids2=DmesonIds,useAnti=True, outputFileName=outputDir+"DDbar_correlation"),
        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[10,15],pTCut2=[2,5],ids1=DmesonIds,ids2=DmesonIds,useAnti=True, outputFileName=outputDir+"DDbar_correlation"),
        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[8,12],pTCut2=[4,6],ids1=DmesonIds,ids2=DmesonIds,useAnti=True, outputFileName=outputDir+"DDbar_correlation"),
        
        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[4,8],pTCut2=[10,15],etaCut1=[-4,4],etaCut2=[-2,2],ids1={421,-421},ids2={22,-22}, outputFileName=outputDir+"D0photon_correlation"),

        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[4,8],pTCut2=[10,15],etaCut1=[-4,4],etaCut2=[-2,2],ids1=DmesonIds,ids2={22,-22}, outputFileName=outputDir+"Dphoton_correlation"),
        
        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[2,4],pTCut2=[10,15],etaCut1=[-4,4],etaCut2=[-2,2],ids1=DmesonIds,ids2={22,-22}, outputFileName=outputDir+"Dphoton_correlation"),
        
        #CorrelationYieldAnalysis(pThatBins=pThatPair, etaBins=list([-4+0.1*x for x in range(81)]), phiBins=list([-3.2+0.1*x for x in range(65)]),
        #pTCut1=[2,4],pTCut2=[4,8],etaCut1=[-4,4],etaCut2=[-2,2],ids1=DmesonIds,ids2={22,-22}, outputFileName=outputDir+"Dphoton_correlation"),

        #MomentumFractionAnalysis(pThatBins=pThatPair, pTFractionBins=list([0.1*x for x in range(101)]),
        #pTCut1=[10,15],pTCut2=[0,10000],ids2=DmesonIds,ids1={22,-22}, outputFileName=outputDir+"Dphoton_momentumFraction"),
        
        #MomentumFractionAnalysis(pThatBins=pThatPair, pTFractionBins=list([0.1*x for x in range(101)]),
        #pTCut1=[4,8],pTCut2=[0,10000],ids2=DmesonIds,ids1={22,-22}, outputFileName=outputDir+"Dphoton_momentumFraction"),

        #MomentumFractionAnalysis(pThatBins=pThatPair, pTFractionBins=list([0.1*x for x in range(101)]),
        #pTCut1=[10,15],pTCut2=[0,10000],ids2=DmesonIds,ids1={22,-22},deltaPhiCut=[4*math.pi/5,math.pi], outputFileName=outputDir+"Dphoton_momentumFraction"),
        
        #MomentumFractionAnalysis(pThatBins=pThatPair, pTFractionBins=list([0.1*x for x in range(101)]),
        #pTCut1=[4,8],pTCut2=[0,10000],ids2=DmesonIds,ids1={22,-22},deltaPhiCut=[4*math.pi/5,math.pi], outputFileName=outputDir+"Dphoton_momentumFraction"),
        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]),
        #                 ids=chargeHadronId, outputFileName=outputDir+"ch_eta_yield"),
        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]), useRap=True,
        #                 ids=chargeHadronId, outputFileName=outputDir+"ch_rap_yield"),
        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]), pTCut = [90, 100],
        #                 ids=chargeHadronId, outputFileName=outputDir+"ch_eta_yield"),
        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]), useRap=True, pTCut=[90, 100],
        #                 ids=chargeHadronId, outputFileName=outputDir+"ch_rap_yield"),

        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]),
        #                 ids=[421, -421], outputFileName=outputDir+"D0_eta_yield"),
        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]), useRap=True,
        #                 ids=[421, -421], outputFileName=outputDir+"D0_rap_yield"),
        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]), pTCut = [90, 100],
        #                 ids=[421, -421], outputFileName=outputDir+"D0_eta_yield"),
        #etaYieldAnalysis(pThatBins=pThatPair, etaBins=list([-5+0.1*x for x in range(101)]), useRap=True, pTCut=[90, 100],
        #                 ids=[421, -421], outputFileName=outputDir+"D0_rap_yield"),
        
        # FlowAnalysis(pThatBins=pThatPair, pTBins=[2, 3, 4, 5, 6, 8, 10, 15, 20, 40],
        #             ids=[411, -411, 421, -421, 413, -413, 423, -423], rapidityCut=[-1, 1], outputFileName=outputDir+"D0_v2"),

        #InclusiveJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[100,112,125,141,158,177,199,223,251,281,316,354,398,501,630,999],
        #                            jetRadius=0.2, jetpTMin=1, jetRapidityCut=[-2.8, 2.8],
        #                            outputFileName=outputDir+"inclusiveJet_yield"),

        InclusiveJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[100,112,125,141,158,177,199,223,251,281,316,354,398,501,630,999],
                                    jetRadius=0.3, jetpTMin=1, jetRapidityCut=[-2.8, 2.8],
                                    outputFileName=outputDir+"inclusiveJet_yield"),

        InclusiveJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[100,112,125,141,158,177,199,223,251,281,316,354,398,501,630,999],
                                    jetRadius=0.4, jetpTMin=1, jetRapidityCut=[-2.8, 2.8],
                                    outputFileName=outputDir+"inclusiveJet_yield"),
        
        #InclusiveJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[100,112,125,141,158,177,199,223,251,281,316,354,398,501,630,999],
        #                            jetRadius=0.5, jetpTMin=1, jetRapidityCut=[-2.8, 2.8],
        #                            outputFileName=outputDir+"inclusiveJet_yield"),

        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        #usepT=False,jetRadius=0.4,jetpTMin=126,jetpTMax=158,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        #usepT=False,jetRadius=0.4,jetpTMin=158,jetpTMax=200,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        #usepT=False,jetRadius=0.4,jetpTMin=200,jetpTMax=251,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        #usepT=False,jetRadius=0.4,jetpTMin=251,jetpTMax=316,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        #usepT=False,jetRadius=0.4,jetpTMin=316,jetpTMax=398,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),

        JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        usepT=False,jetRadius=0.4,jetpTMin=126,jetpTMax=158,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        usepT=False,jetRadius=0.4,jetpTMin=158,jetpTMax=200,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        usepT=False,jetRadius=0.4,jetpTMin=200,jetpTMax=251,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        #usepT=False,jetRadius=0.4,jetpTMin=251,jetpTMax=316,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[0.00794328,0.01,0.0125893,0.158489,0.199526,0.0251189,0.0316228,0.0398107,0.0501187,0.0630957,0.0794328,0.1,0.125893,0.158489,0.199526,0.251189,0.316228,0.398107,0.501187,0.630958,0.794329,1],
        #usepT=False,jetRadius=0.4,jetpTMin=316,jetpTMax=398,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),

        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        #usepT=True,jetRadius=0.4,jetpTMin=126,jetpTMax=158,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        #usepT=True,jetRadius=0.4,jetpTMin=158,jetpTMax=200,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        #usepT=True,jetRadius=0.4,jetpTMin=200,jetpTMax=251,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        #usepT=True,jetRadius=0.4,jetpTMin=251,jetpTMax=316,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        #usepT=True,jetRadius=0.4,jetpTMin=316,jetpTMax=398,jetRapidityCut=[-2.1, 2.1],outputFileName=outputDir+"fragmentation_function"),

        JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        usepT=True,jetRadius=0.4,jetpTMin=126,jetpTMax=158,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        usepT=True,jetRadius=0.4,jetpTMin=158,jetpTMax=200,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        usepT=True,jetRadius=0.4,jetpTMin=200,jetpTMax=251,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        #usepT=True,jetRadius=0.4,jetpTMin=251,jetpTMax=316,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        #JetFragmentationFunctionAnalysis(pThatBins=pThatPair,bins=[1,1.3318,1.7758,2.3677,3.1569,4.2092,5.6123,7.4831,9.9775,13.3033,17.7377,23.6503,31.5337,42.0449,56.0599,74.7466,99.6621],
        #usepT=True,jetRadius=0.4,jetpTMin=316,jetpTMax=398,jetRapidityCut=[-0.3, 0.3],outputFileName=outputDir+"fragmentation_function"),
        #HeavyJetpTYieldAnalysis(pThatBins=pThatPair, pTBins=[40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360],
        #                        jetRadius=0.2, jetEtaCut=[-2.1, 2.1], drCut=0.3, heavypTCut=[5, 30000], ids=[411, -411, 421, -421, 413, -413, 423, -423], outputFileName=outputDir+"DJet_yield"),
        #HeavyRadialProfileAnalysis(pThatBins=pThatPair, rBins=[0, 0.05, 0.1, 0.3, 0.5], jetRadius=0.3, jetpTMin=60, jetEtaCut=[-1.6, 1.6], heavypTCut=[20, 30000], heavyEtaCut=[-2, 2], ids=[-421, 421],
        #                           outputFileName=outputDir+"D0_jet_radialprofile"),
        #HeavyRadialProfileAnalysis(pThatBins=pThatPair, rBins=[0, 0.05, 0.1, 0.3, 0.5], jetRadius=0.3, jetpTMin=60, jetEtaCut=[-1.6, 1.6], heavypTCut=[4, 20], heavyEtaCut=[-2, 2], ids=[-421, 421],
         #                          outputFileName=outputDir+"D0_jet_radialprofile"),
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
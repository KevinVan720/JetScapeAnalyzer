from operator import truediv
import numpy as np
import h5py
import random
import os
import sys
import fnmatch
import re
import math
from itertools import groupby
import fastjet as fj
import fjext
from time import time
#from pdxml_reader import *

logDelta=0.000000001

chargeHadronIds={-212212,
 -212112,
 -202212,
 -202112,
 -100211,
 -5332,
 -5222,
 -5132,
 -5112,
 -4324,
 -4322,
 -4232,
 -4224,
 -4222,
 -4214,
 -4212,
 -4122,
 -3334,
 -3314,
 -3312,
 -3224,
 -3222,
 -3114,
 -3112,
 -2224,
 -2214,
 -2212,
 -1114,
 -541,
 -521,
 -433,
 -431,
 -413,
 -411,
 -323,
 -321,
 -213,
 -211,
 -24,
 -15,
 -13,
 -11,
 11,
 13,
 15,
 24,
 211,
 213,
 321,
 323,
 411,
 413,
 431,
 433,
 521,
 541,
 1114,
 2212,
 2214,
 2224,
 3112,
 3114,
 3222,
 3224,
 3312,
 3314,
 3334,
 4122,
 4212,
 4214,
 4222,
 4224,
 4232,
 4322,
 4324,
 5112,
 5132,
 5222,
 5332,
 100111,
 100211,
 202112,
 202212,
 212112,
 212212,
 1000010010,
 1000010020,
 1000010030,
 1000020030,
 1000020040,
 1000030070,
 1000040090,
 1000060120,
 1000060130,
 1000070140,
 1000080160,
 1000080180,
 1000100210,
 1000100220,
 1000110050,
 1000160330,
 1000180400,
 1000280560,
 1000541280,
 1000822080,
 1000862220}

'''
chargeHadronId = [211, 213, 9000211, 10213, 20213, 100211, 215, 9000213, 10211, 100213, 9010213, 10215,
 217, 30213, 9010211, 219, 321, 323, 10323, 20323, 100323, 10321, 325, 30323, 10325, 327, 20325, 329, 2212,
  12212, 2124, 22212, 32212, 2216, 12216, 22124, 42212, 32124, 2128, 1114, 2214, 2224, 31114, 32214, 32224,
   1112, 2122, 2222, 11114, 12214, 12224, 1116, 2126, 2226, 21112, 22122, 22222, 21114, 22214, 22224, 11116,
    12126, 12226, 1118, 2218, 2228, 3222, 3112, 3114, 3224, 13112, 13222, 13114, 13224, 23112, 23222, 3116,
     3226, 13116, 13226, 23114, 23224, 3118, 3228, 3312, 3314, 203312, 13314, 103316, 203316, 3334, 203338,
     -211, -213, -9000211, -10213, -20213, -100211, -215, -9000213, -10211, -100213, -9010213, -10215, -217,
     -30213, -9010211, -219, -321, -323, -10323, -20323, -100323, -10321, -325, -30323, -10325, -327, -20325,
      -329, -2212, -12212, -2124, -22212, -32212, -2216, -12216, -22124, -42212, -32124, -2128, -1114, -2214,
       -2224, -31114, -32214, -32224, -1112, -2122, -2222, -11114, -12214, -12224, -1116, -2126, -2226, -21112,
        -22122, -22222, -21114, -22214, -22224, -11116, -12126, -12226, -1118, -2218, -2228, -3222, -3112, -3114,
         -3224, -13112, -13222, -13114, -13224, -23112, -23222, -3116, -3226, -13116, -13226, -23114, -23224, -3118,
          -3228, -3312, -3314, -203312, -13314, -103316, -203316, -3334, -203338,
          411,-411,413,-413, 521,-521,523,-523]
'''


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    Neffective = np.sum(weights)**2/np.sum(weights**2)
    return (average, np.sqrt(variance/(Neffective-1)))

def strNotContain(name):
    noNames = ["File", "pThat", "Count",
              "jetDefinition", "jetSelector", "Q0", "Q2", "Hydro"]
    for n in noNames:
        if n.upper() in name.upper() or n.lower() in name.lower():
            return False
    return True

def findIndex(valList, val):
    assert(len(valList) >= 2)
    for i in range(len(valList)-1):
        if val > valList[i] and val <= valList[i+1]:
            return i
    return -1

def withinInterval(y, interval):
    if interval == None or len(interval)==0:
        return True
    if len(interval) == 2:
        assert(interval[0] < interval[1])
        if y >= interval[0] and y < interval[1]:
            return True
    return False


#@jit(nopython=True)
def rap(a, b):
    return 0.5 * np.log((a + b + logDelta) /
                        (abs(a - b) + logDelta))

#currently only supports status code of 0 and -1!!!
def saveParticleInfo(pid,status):
    sign=pid/abs(pid)
    
    return (abs(pid)*100+abs(status))*sign

def loadParticleInfo(a):
    sign=int(a/abs(a))
    base=(abs(a)//100)*100
    status=-(abs(a)-base)
    pid=(a-status)/100
    return (int(pid),int(status))

class Particle:
    def __init__(self, pid, status, E, px, py, pz):
        self.pid = pid
        self.status = status
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz
        self.pT = np.sqrt(px**2+py**2)
        self.y = rap(E,pz)
        self.eta = rap(np.sqrt(px**2+py**2+pz**2), pz)
        self.phi = np.arctan2(py, px)


    def __eq__(self, other): 
        if not isinstance(other, Particle):
            # don't attempt to compare against unrelated types
            return False

        return self.pid == other.pid and self.status == other.status and self.E == other.E and self.px == other.px and self.py == other.py and self.pz == other.pz


class JetScapeReader:
    def __init__(self, fileName, headerName=None, pTMin=0.1):
        self.fileName = fileName
        self.headerName = headerName

        self.pTMin=pTMin
        self.currentEventCount = 0
        self.currentCrossSection = 0
        self.currentHydroInfo = []
        self.particleList = []

    def parseEventHeader(self, strlist):
        '''
        currently only support a single header format
        with event cross section and hydro flow information
        '''
        if len(strlist) == 25:
            assert(
                "sigmaGen" in strlist and "Ncoll" in strlist and "v_2" in strlist and "psi_2" in strlist)
            self.currentCrossSection = float(strlist[6].strip())
            self.currentHydroInfo = [float(a.strip()) for a in strlist[8::2]]
            self.currentEventCount += 1
        
        elif len(strlist) == 9 and "sigmaGen" in strlist and "Ncoll" in strlist:
            #assert(
            #    "sigmaGen" in strlist and "Ncoll" in strlist)
            self.currentCrossSection = float(strlist[6].strip())
            self.currentEventCount += 1
        else:
            #old event header with no cross section
            self.currentEventCount += 1

    def readAllEvents(self):
        if self.headerName!=None:
            hf=open(self.headerName, "r", 32768)
        with open(self.fileName, "r", 32768) as f:
            for line in f:
                strlist = line.rstrip().split()
                if line.startswith("#"):   # A new event
                    if len(self.particleList) == 0:
                        if self.headerName!=None:
                            headerlist=hf.readline().rstrip().split()
                            #print(headerlist)
                            self.parseEventHeader(headerlist)
                        else:
                            self.parseEventHeader(strlist)
                        continue

                    yield self.particleList
                    if self.headerName!=None:
                        headerlist=hf.readline().rstrip().split()
                        self.parseEventHeader(headerlist)
                        #print(headerlist)

                    else:
                        self.parseEventHeader(strlist)
                    #self.parseEventHeader(strlist)
                    self.particleList.clear()

                else:
                    if len(strlist) == 9:
                        i, pid, status, E, px, py, pz, eta, phi = [
                            float(a) for a in strlist
                        ]
                        if np.sqrt(px**2+py**2)>self.pTMin:
                            self.particleList.append(
                                Particle(pid, status, E, px, py, pz)
                            )
                    elif len(strlist) == 6:
                        pid, status, E, px, py, pz = [
                            float(a) for a in strlist
                        ]
                        if np.sqrt(px**2+py**2)>self.pTMin:
                            self.particleList.append(
                                Particle(pid, status, E, px, py, pz))
                    else:
                        print("Unknown particle length")


def getCrossSectionFromFile(inputDir, fileName):
    CSFile = [file for file in os.listdir(inputDir) if fnmatch.fnmatch(
        file, fileName)]
    if len(CSFile) > 0:
        sigma = np.loadtxt(inputDir+CSFile[0])[0]
        return sigma
    else:
        raise FileNotFoundError()


'''
Assumption: 
1.the simulation is setup with many pThatBins and we want to combine them in the analysis
2. the simulation is interested in particles pf certain id and status
'''

class AnalysisBase:
    def __init__(self, pThatBins=[], ids=set(), status=[], outputFileName=""):
        self.outputFileName = outputFileName + \
            "_%04d" % random.randint(
                0, 9999)+str(hash(type(self).__name__) % 1000)+".txt"

        self.ids = ids
        self.status = status

        self.pThatBins = pThatBins
        self.NpThatBins = len(self.pThatBins)-1

        self.pThatEventCounts = [0 for i in range(self.NpThatBins)]
        self.pThatEventCrossSections = [
            0 for i in range(self.NpThatBins)]

    def setCrossSection(self, pThatIndex, crossSection):
        self.pThatIndex = pThatIndex
        self.pThatEventCrossSections[self.pThatIndex] = crossSection
        self.pThatEventCounts[self.pThatIndex] += 1

    def setStatus(self, pThatIndex, reader):
        self.pThatIndex = pThatIndex
        self.pThatEventCrossSections[self.pThatIndex] = reader.currentCrossSection
        self.pThatEventCounts[self.pThatIndex] += 1

    def analyzeEvent(self, particles):
        raise NotImplementedError()

    def outputHeader(self):
        rst = ""
        for (name, val) in vars(self).items():
            if strNotContain(name):
                rst += name+" "+str(val)+"\n"
        return rst

    def outputResult(self):
        raise NotImplementedError()


class JetAnalysisBase(AnalysisBase):
    def __init__(self, jetRadius=0.4, jetpTMin=1, jetpTMax=None, jetRapidityCut=None, jetEtaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.clusterPower = -1
        self.jetRadius = jetRadius
        self.jetpTMin = jetpTMin
        self.jetpTMax=jetpTMax
        self.jetRapidityCut = jetRapidityCut
        self.jetEtaCut = jetEtaCut

        if self.clusterPower == -1:
            self.jetDefinition = fj.JetDefinition(
                fj.antikt_algorithm, self.jetRadius)

        self.jetSelector = fj.SelectorPtMin(self.jetpTMin)
        if self.jetpTMax!= None:
            self.jetSelector = self.jetSelector & fj.SelectorPtMax(
                self.jetpTMax)
        if self.jetRapidityCut != None:
            self.jetSelector = self.jetSelector & fj.SelectorRapRange(
                self.jetRapidityCut[0], self.jetRapidityCut[1])
        if self.jetEtaCut != None:
            self.jetSelector = self.jetSelector & fj.SelectorEtaRange(
                self.jetEtaCut[0], self.jetEtaCut[1])

    def fillFastJetConstituents(self, hadrons):

        # Create a vector of fastjet::PseudoJets from arrays of px,py,pz,e
        #fj_particles = fjext.vectorize_px_py_pz_e(px, py, pz, e)
        fj_particles = [fj.PseudoJet(
            hadron.px, hadron.py, hadron.pz, hadron.E) for hadron in hadrons]
        for i in range(len(fj_particles)):
            fj_particles[i].set_user_index(int(saveParticleInfo(hadrons[i].pid, hadrons[i].status)))

        return fj_particles

class etaYieldAnalysis(AnalysisBase):
    def __init__(self, etaBins=[], useRap=False, pTCut=None, **kwargs):
        super().__init__(**kwargs)
        self.etaBins = etaBins
        self.NetaBins = len(self.etaBins)-1

        self.useRap=useRap

        self.pTCut = pTCut
        self.countStorage = [
            [0 for j in range(self.NetaBins)] for i in range(self.NpThatBins)]

    def analyzeEvent(self, particles):

        # filtering the particles first to save time
        particles = [p for p in particles if p.pid in self.ids and withinInterval(
            p.pT, self.pTCut)]
        for p in particles:
            if self.useRap:
                i = findIndex(self.etaBins, p.y)
            else:
                i = findIndex(self.etaBins, p.eta)
            if i >= 0:
                self.countStorage[self.pThatIndex][i] += 1

    def outputResult(self):
        if np.sum(self.pThatEventCounts) == 0:
            return

        rst = [0 for j in range(self.NetaBins)]
        err = [0 for j in range(self.NetaBins)]
        for pThat in range(self.NpThatBins):
            for eta in range(self.NetaBins):
                if self.pThatEventCounts[pThat] > 0:
                    normalizeFactor = self.pThatEventCounts[pThat]*(
                        self.etaBins[eta+1]-self.etaBins[eta])
                    rst[eta] += self.countStorage[pThat][eta] * \
                        self.pThatEventCrossSections[pThat]/normalizeFactor
                    err[eta] += self.countStorage[pThat][eta] * \
                        self.pThatEventCrossSections[pThat]**2 / \
                        normalizeFactor**2

        err = [np.sqrt(x) for x in err]
        etaBinsAvg = (np.array(self.etaBins[0:-1])+np.array(self.etaBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose(
            [etaBinsAvg, rst, err]), header=self.outputHeader())

class pTYieldAnalysis(AnalysisBase):
    def __init__(self, pTBins=[], pTMin=0.01, rapidityCut=None, etaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.pTBins = pTBins
        self.NpTBins = len(self.pTBins)-1

        self.pTMin = pTMin
        self.rapidityCut = rapidityCut
        self.etaCut = etaCut
        self.countStorage = [
            [0 for j in range(self.NpTBins)] for i in range(self.NpThatBins)]

    def analyzeEvent(self, particles):

        # filtering the particles first to save time
        particles = [p for p in particles if p.pid in self.ids and p.pT > self.pTMin and withinInterval(
            p.eta, self.etaCut) and withinInterval(p.y, self.rapidityCut)]
        for p in particles:
            i = findIndex(self.pTBins, p.pT)

            if i >= 0:
                if p.status>=0:
                    self.countStorage[self.pThatIndex][i] += 1
                if p.status<0 and self.countStorage[self.pThatIndex][i]>0:
                    self.countStorage[self.pThatIndex][i] -= 1

    def outputResult(self):
        if np.sum(self.pThatEventCounts) == 0:
            return

        rst = [0 for j in range(self.NpTBins)]
        err = [0 for j in range(self.NpTBins)]
        for pThat in range(self.NpThatBins):
            for pT in range(self.NpTBins):
                if self.pThatEventCounts[pThat] > 0:
                    normalizeFactor = self.pThatEventCounts[pThat]*(
                        self.pTBins[pT+1]-self.pTBins[pT])*(self.pTBins[pT+1]+self.pTBins[pT])
                    rst[pT] += self.countStorage[pThat][pT] * \
                        self.pThatEventCrossSections[pThat]/normalizeFactor
                    err[pT] += self.countStorage[pThat][pT] * \
                        self.pThatEventCrossSections[pThat]**2 / \
                        normalizeFactor**2

        err = [np.sqrt(x) for x in err]
        ptBinsAvg = (np.array(self.pTBins[0:-1])+np.array(self.pTBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose(
            [ptBinsAvg, rst, err]), header=self.outputHeader())

class CorrelationYieldAnalysis(AnalysisBase):
    def __init__(self, ids1=set(), ids2=set(), etaBins=[],phiBins=[], useAnti=False, useRap=False, pTCut1=None, pTCut2=None,rapidityCut1=None, etaCut1=None,rapidityCut2=None, etaCut2=None, **kwargs):
        super().__init__(**kwargs)
        self.ids1=ids1
        self.ids2=ids2
        self.etaBins = etaBins
        self.NetaBins = len(self.etaBins)-1
        self.phiBins = phiBins
        self.NphiBins = len(self.phiBins)-1

        self.useRap=useRap
        self.useAnti=useAnti

        self.pTCut1 = pTCut1
        self.pTCut2 = pTCut2
        self.rapidityCut1 = rapidityCut1
        self.etaCut1 = etaCut1
        self.rapidityCut2 = rapidityCut2
        self.etaCut2 = etaCut2
        self.countStorage = [[
            [0 for j in range(self.NetaBins)] for i in range(self.NphiBins)] for k in range(self.NpThatBins)]

    def analyzeEvent(self, particles):

        # filtering the particles first to save time
        if self.useRap:
            particles1 = [p for p in particles if p.pid in self.ids1 and withinInterval(
            p.pT, self.pTCut1) and withinInterval(
            p.y, self.rapidityCut1)]
            particles2 = [p for p in particles if p.pid in self.ids1 and withinInterval(
            p.pT, self.pTCut2) and withinInterval(
            p.y, self.rapidityCut2)]
        else:
            particles1 = [p for p in particles if p.pid in self.ids1 and withinInterval(
            p.pT, self.pTCut1) and withinInterval(
            p.eta, self.etaCut1)]
            particles2 = [p for p in particles if p.pid in self.ids2 and withinInterval(
            p.pT, self.pTCut2) and withinInterval(
            p.eta, self.etaCut2)]
            
        
        for p1 in particles1:
            for p2 in particles2:
                if p1!=p2 and (not self.useAnti or p1.pid*p2.pid<0):                
                    if self.useRap:
                        j= findIndex(self.etaBins, p1.y-p2.y)
                    else:
                        j = findIndex(self.etaBins, p1.eta-p2.eta)
                    i= findIndex(self.phiBins, p1.phi-p2.phi)
  
                    if i >= 0 and j>=0 :
                        self.countStorage[self.pThatIndex][i][j] += 1

    def outputResult(self):
        if np.sum(self.pThatEventCounts) == 0:
            return

        rst = [[0 for j in range(self.NetaBins)] for i in range(self.NphiBins)]
        #err = [0 for j in range(self.NetaBins)]
        for pThat in range(self.NpThatBins):
            for phi in range(self.NphiBins):
                for eta in range(self.NetaBins):
                    if self.pThatEventCounts[pThat] > 0:
                        normalizeFactor = self.pThatEventCounts[pThat]*(
                            self.etaBins[eta+1]-self.etaBins[eta])*(
                            self.phiBins[phi+1]-self.phiBins[phi])
                        rst[phi][eta] += self.countStorage[pThat][phi][eta] * \
                            self.pThatEventCrossSections[pThat]/normalizeFactor
                        #err[eta] += self.countStorage[pThat][eta] * \
                        #self.pThatEventCrossSections[pThat]**2 / \
                        #normalizeFactor**2

        #err = [np.sqrt(x) for x in err]
        #etaBinsAvg = (np.array(self.etaBins[0:-1])+np.array(self.etaBins[1:]))/2
        
        np.savetxt(self.outputFileName, rst, header=self.outputHeader())
        
class MomentumFractionAnalysis(AnalysisBase):
    def __init__(self, ids1=set(), ids2=set(), pTFractionBins=[], useRap=False, pTCut1=None, pTCut2=None,rapidityCut1=None, etaCut1=None,rapidityCut2=None, etaCut2=None,deltaPhiCut=None, **kwargs):
        super().__init__(**kwargs)
        self.ids1=ids1
        self.ids2=ids2
 
        self.pTFractionBins = pTFractionBins
        self.NpTFractionBins = len(self.pTFractionBins)-1

        self.useRap=useRap

        self.pTCut1 = pTCut1
        self.pTCut2 = pTCut2
        self.rapidityCut1 = rapidityCut1
        self.etaCut1 = etaCut1
        self.rapidityCut2 = rapidityCut2
        self.etaCut2 = etaCut2
        if deltaPhiCut:
            self.deltaPhiCut=deltaPhiCut
        else:
            self.deltaPhiCut=[0,math.pi]
        self.countStorage = [[
            0 for i in range(self.NpTFractionBins)] for k in range(self.NpThatBins)]

    def analyzeEvent(self, particles):

        # filtering the particles first to save time
        if self.useRap:
            particles1 = [p for p in particles if p.pid in self.ids1 and withinInterval(
            p.pT, self.pTCut1) and withinInterval(
            p.y, self.rapidityCut1)]
            particles2 = [p for p in particles if p.pid in self.ids1 and withinInterval(
            p.pT, self.pTCut2) and withinInterval(
            p.y, self.rapidityCut2)]
        else:
            particles1 = [p for p in particles if p.pid in self.ids1 and withinInterval(
            p.pT, self.pTCut1) and withinInterval(
            p.eta, self.etaCut1)]
            particles2 = [p for p in particles if p.pid in self.ids2 and withinInterval(
            p.pT, self.pTCut2) and withinInterval(
            p.eta, self.etaCut2)]
            
        
        for p1 in particles1:
            for p2 in particles2:
                deltaPhi=min(abs(p1.phi-p2.phi),2*math.pi-abs(p1.phi-p2.phi))
                if p1!=p2 and deltaPhi>=self.deltaPhiCut[0] and deltaPhi<=self.deltaPhiCut[1]:                
                    i= findIndex(self.pTFractionBins, p1.pT/p2.pT)
  
                    if i >= 0:
                        self.countStorage[self.pThatIndex][i] += 1

    def outputResult(self):
        if np.sum(self.pThatEventCounts) == 0:
            return

        rst = [0 for j in range(self.NpTFractionBins)]
        err = [0 for j in range(self.NpTFractionBins)]
        for pThat in range(self.NpThatBins):
            for pTFraction in range(self.NpTFractionBins):
                    if self.pThatEventCounts[pThat] > 0:
                        normalizeFactor = self.pThatEventCounts[pThat]*(
                            self.pTFractionBins[pTFraction+1]-self.pTFractionBins[pTFraction])
                        rst[pTFraction] += self.countStorage[pThat][pTFraction] * \
                            self.pThatEventCrossSections[pThat]/normalizeFactor
                        err[pTFraction] += self.countStorage[pThat][pTFraction] * \
                        self.pThatEventCrossSections[pThat]**2 / \
                        normalizeFactor**2

        err = [np.sqrt(x) for x in err]
        pTBinsAvg = (np.array(self.pTFractionBins[0:-1])+np.array(self.pTFractionBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose(
            [pTBinsAvg, rst, err]), header=self.outputHeader())

class JetShapeAnalysis(JetAnalysisBase):
    def __init__(self, rBins, trackpTMin=0.7,trackpTMax=300, **kwargs):
        super().__init__(**kwargs)
        self.rBins = rBins
        self.NrBins = len(self.rBins)-1
        self.trackpTMin=trackpTMin
        self.trackpTMax=trackpTMax

        self.countStorage = [
            [0 for j in range(self.NrBins)] for i in range(self.NpThatBins)]

    def analyzeEvent(self, particles):

        holes=[x for x in particles if x.status<0]
        particles=[x for x in particles if x.status>=0]

        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        for jet in jets_selected:
            for hadron in particles:
                dr = np.sqrt((hadron.eta-jet.eta())**2 +
                             (hadron.phi-jet.phi())**2)
                i = findIndex(self.rBins, dr)
                if i > 0 and hadron.pT>self.trackpTMin and hadron.pT<self.trackpTMax:
                    self.countStorage[self.pThatIndex][i] += hadron.pt()
            for hadron in holes:
                dr = np.sqrt((hadron.eta-jet.eta())**2 +
                             (hadron.phi-jet.phi())**2)
                i = findIndex(self.rBins, dr)
                if i > 0 and hadron.pT>self.trackpTMin and hadron.pT<self.trackpTMax:
                    self.countStorage[self.pThatIndex][i] -= hadron.pt()

    def outputResult(self):
        if np.sum(self.pThatEventCounts) == 0:
            return

        rst = [0 for j in range(self.NrBins)]
        err = [0 for j in range(self.NrBins)]
        for pThat in range(self.NpThatBins):
            for pT in range(self.NrBins):
                if self.pThatEventCounts[pThat] > 0:
                    normalizeFactor = self.pThatEventCounts[pThat]
                    rst[pT] += self.countStorage[pThat][pT] * \
                        self.pThatEventCrossSections[pThat]/normalizeFactor
                    err[pT] += self.countStorage[pThat][pT] * \
                        self.pThatEventCrossSections[pThat]**2 / \
                        normalizeFactor**2

        err = [np.sqrt(x) for x in err]
        ptBinsAvg = (np.array(self.rBins[0:-1])+np.array(self.rBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose(
            [ptBinsAvg, rst, err]), header=self.outputHeader())

class JetFragmentationFunctionAnalysis(JetAnalysisBase):
    def __init__(self, bins, usepT=True, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.NBins = len(self.bins)-1

        self.usepT=usepT

        self.countStorage = [
            [0 for j in range(self.NBins)] for i in range(self.NpThatBins)]

    def analyzeEvent(self, particles):

        holes=[x for x in particles if x.status<0]
        particles=[x for x in particles if x.status>=0]

        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        Njet=len(jets_selected)

        for jet in jets_selected:
            constituents = jet.constituents()
            for hadron in constituents:
                pid, status=loadParticleInfo(hadron.user_index())
                dr = np.sqrt((hadron.eta()-jet.eta())**2 +
                             (hadron.phi()-jet.phi())**2)
                z=hadron.pt()*np.cos(dr)/jet.pt()
                if self.usepT:
                    i = findIndex(self.bins, hadron.pt())
                else:
                    i = findIndex(self.bins, z)
                if i > 0:
                    diff=self.bins[i+1]-self.bins[i]
                    self.countStorage[self.pThatIndex][i] += 1/diff/Njet

            for hole in holes:
                dr = np.sqrt((hole.eta-jet.eta())**2 +
                             (hole.phi-jet.phi())**2)
                z=hole.pT*np.cos(dr)/jet.pt()
                if self.usepT:
                    i = findIndex(self.bins, hole.pT)
                else:
                    i = findIndex(self.bins, z)
                if i > 0:
                    diff=self.bins[i+1]-self.bins[i]
                    self.countStorage[self.pThatIndex][i] -= 1/diff/Njet



    def outputResult(self):
        if np.sum(self.pThatEventCounts) == 0:
            return

        rst = [0 for j in range(self.NBins)]
        err = [0 for j in range(self.NBins)]
        for pThat in range(self.NpThatBins):
            for i in range(self.NBins):
                if self.pThatEventCounts[pThat] > 0:
                    normalizeFactor = self.pThatEventCounts[pThat]
                    rst[i] += self.countStorage[pThat][i] * \
                        self.pThatEventCrossSections[pThat]/normalizeFactor
                    err[i] += self.countStorage[pThat][i] * \
                        self.pThatEventCrossSections[pThat]**2 / \
                        normalizeFactor**2

        err = [np.sqrt(x) for x in err]
        ptBinsAvg = (np.array(self.bins[0:-1])+np.array(self.bins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose(
            [ptBinsAvg, rst, err]), header=self.outputHeader())


class InclusiveJetpTYieldAnalysis(JetAnalysisBase, pTYieldAnalysis):
    def __init__(self, leadingpTCut=0, **kwargs):
        super().__init__(**kwargs)
        self.leadingpTCut = leadingpTCut

    def analyzeEvent(self, particles):

        holes=[x for x in particles if x.status<0]
        particles=[x for x in particles if x.status>=0]

        fjHadrons = self.fillFastJetConstituents(particles)

        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        for jet in jets_selected:
            jetpT=jet.pt()

            considered=False
            for p in particles:
                dr = np.sqrt((p.eta-jet.eta())**2 +
                             (p.phi-jet.phi())**2)
                if dr<=self.jetRadius and p.pid in chargeHadronIds and p.pT>self.leadingpTCut:
                    considered=True
                    break

            for hole in holes:
                dr = np.sqrt((hole.eta-jet.eta())**2 +
                             (hole.phi-jet.phi())**2)
                if dr<=self.jetRadius:
                    jetpT-=hole.pT

            i = findIndex(self.pTBins, jetpT)
            if i >= 0 and considered:
                self.countStorage[self.pThatIndex][i] += 1


class HeavyJetpTYieldAnalysis(JetAnalysisBase, pTYieldAnalysis):
    def __init__(self, drCut=0.3, heavypTCut=None, heavyRapidityCut=None, heavyEtaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.drCut = 0.3
        self.heavypTCut = heavypTCut
        self.heavyRapidityCut = heavyRapidityCut
        self.heavyEtaCut = heavyEtaCut

    def analyzeEvent(self, particles):

        holes=[x for x in particles if x.status<0]
        particles=[x for x in particles if x.status>=0]

        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        fjHadrons = [hadron for hadron in fjHadrons if loadParticleInfo(hadron.user_index())[0] in self.ids
                     and withinInterval(hadron.pt(), self.heavypTCut)
                     and withinInterval(hadron.eta(), self.heavyEtaCut)
                     and withinInterval(hadron.rap(), self.heavyRapidityCut)]

        for jet in jets_selected:
            jetpT=jet.pt()
            constituents = jet.constituents()
            for hole in holes:
                dr = np.sqrt((hole.eta-jet.eta())**2 +
                             (hole.phi-jet.phi())**2)
                if dr<=self.jetRadius:
                    jetpT-=hole.pT

            for hadron in fjHadrons:
                dr = np.sqrt((hadron.eta()-jet.eta())**2 +
                             (hadron.phi()-jet.phi())**2)
                if dr < self.drCut:
                    i = findIndex(self.pTBins, jetpT)
                    if i >= 0:
                        self.countStorage[self.pThatIndex][i] += 1
                        break

class HeavyRadialProfileAnalysis(JetShapeAnalysis):
    def __init__(self, heavypTCut=None, heavyRapidityCut=None, heavyEtaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.heavypTCut = heavypTCut
        self.heavyRapidityCut = heavyRapidityCut
        self.heavyEtaCut = heavyEtaCut

    def analyzeEvent(self, particles):

        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        for jet in jets_selected:
            #constituents = jet.constituents()
            for hadron in fjHadrons:
                dr = np.sqrt((hadron.eta()-jet.eta())**2 +
                             (hadron.phi()-jet.phi())**2)
                if hadron.user_index() in self.ids \
                        and withinInterval(hadron.pt(), self.heavypTCut) \
                        and withinInterval(hadron.eta(), self.heavyEtaCut) \
                        and withinInterval(hadron.rap(), self.heavyRapidityCut):
                    i = findIndex(self.rBins, dr)
                    if i >= 0:
                        self.countStorage[self.pThatIndex][i] += 1
                        break


class FlowAnalysis(pTYieldAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.allHydros = []
        self.Q2_Re = [[[] for j in range(
            self.NpTBins)] for i in range(self.NpThatBins)]
        self.Q2_Im = [[[] for j in range(
            self.NpTBins)] for i in range(self.NpThatBins)]
        self.Q0 = [[[] for j in range(
            self.NpTBins)] for i in range(self.NpThatBins)]

    def setStatus(self, pThatIndex, reader):
        super().setStatus(pThatIndex, reader)
        self.currentHydro = reader.currentHydroInfo
        if self.currentHydro not in self.allHydros:

            self.allHydros.append(self.currentHydro)
            for pThat in range(self.NpThatBins):
                for pT in range(self.NpTBins):
                    self.Q2_Re[pThat][pT].append(0)
                    self.Q2_Im[pThat][pT].append(0)
                    self.Q0[pThat][pT].append(0)

    def analyzeEvent(self, particles):
        hydroId = self.allHydros.index(self.currentHydro)

        particles = [p for p in particles if p.pid in self.ids and p.pT > self.pTMin and withinInterval(
            p.eta, self.etaCut) and withinInterval(p.y, self.rapidityCut)]

        for p in particles:
            i = findIndex(self.pTBins, p.pT)
            if i > 0:
                self.Q2_Re[self.pThatIndex][i][hydroId] += np.cos(2 * p.phi)
                self.Q2_Im[self.pThatIndex][i][hydroId] += np.sin(2 * p.phi)
                self.Q0[self.pThatIndex][i][hydroId] += 1

    def outputResult(self):
        if np.sum(self.pThatEventCounts) == 0:
            return

        v2_ch_rms = np.sqrt(np.mean(np.array(self.allHydros)[:, 3] ** 2))
        v2_all = [[] for j in range(self.NpTBins)]
        rst = [0 for j in range(self.NpTBins)]
        for hydroId in range(len(self.allHydros)):
            for pT in range(self.NpTBins):
                temp_Q2_Re = 0
                temp_Q2_Im = 0
                temp_Q0 = 0
                for pThat in range(self.NpThatBins):
                    if self.pThatEventCounts[pThat] == 0:
                        continue
                    temp_Q2_Re += self.Q2_Re[pThat][pT][hydroId] * \
                        self.pThatEventCrossSections[pThat] / \
                        self.pThatEventCounts[pThat]
                    temp_Q2_Im += self.Q2_Im[pThat][pT][hydroId] * \
                        self.pThatEventCrossSections[pThat] / \
                        self.pThatEventCounts[pThat]
                    temp_Q0 += self.Q0[pThat][pT][hydroId] * \
                        self.pThatEventCrossSections[pThat] / \
                        self.pThatEventCounts[pThat]
                if temp_Q0 == 0:
                    continue
                v2_temp = (
                    np.sqrt(temp_Q2_Re ** 2 + temp_Q2_Im ** 2)
                    / temp_Q0
                )
                psi_temp = 0.5 * \
                    np.arctan2(temp_Q2_Im, temp_Q2_Re)
                weight = temp_Q0*self.allHydros[hydroId][0]
                v2_ch = self.allHydros[hydroId][3]
                psi_ch = self.allHydros[hydroId][4]

                v2_all[pT].append(
                    [v2_temp * v2_ch * np.cos(2 * (psi_temp - psi_ch)), weight])

        for i in range(self.NpTBins):
            v2 = np.array(v2_all[i])
            if len(v2) > 0:
                rst[i] = (weighted_avg_and_std(
                    v2[:, 0], v2[:, 1])[0]/v2_ch_rms)

        ptBinsAvg = (np.array(self.pTBins[0:-1])+np.array(self.pTBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose(
            [ptBinsAvg, rst]), header=self.outputHeader())

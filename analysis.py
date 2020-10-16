import numpy as np
import h5py
import random
import os
import sys
import fnmatch
import re
from itertools import groupby
import fastjet as fj
import fjext
from time import time
#from numba import jit
#import time

logDelta=0.000000001

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
               "Bins", "jetDefinition", "jetSelector", "Q0", "Q2", "Hydro"]
    for n in noNames:
        if n.upper() in name.upper() or n.lower() in name.lower():
            return False
    return True

def findIndex(valList, val):
    assert(len(valList) >= 2)
    for i in range(len(valList)-1):
        if val > valList[i] and val < valList[i+1]:
            return i
    return -1

def withinInterval(y, interval):
    if interval == None:
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


class JetScapeReader:
    def __init__(self, fileName, pTMin=0.1):
        self.fileName = fileName

        self.pTMin=pTMin
        self.currentEventCount = 0
        self.currentCrossSection = 0
        self.currentHydroInfo = []
        self.particleList = []

    def readEventHeader(self, strlist):
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
        
        if len(strlist) == 9:
            assert(
                "sigmaGen" in strlist and "Ncoll" in strlist)
            self.currentCrossSection = float(strlist[6].strip())
            self.currentEventCount += 1

    def readAllEvents(self):
        with open(self.fileName, "r", 32768) as f:
            for line in f:
                strlist = line.rstrip().split()
                if line.startswith("#"):   # A new event
                    if len(self.particleList) == 0:
                        self.readEventHeader(strlist)
                        continue

                    yield self.particleList
                    self.readEventHeader(strlist)
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
    def __init__(self, pThatBins=[], ids=[], status=[], outputFileName=""):
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
    def __init__(self, jetRadius=0.4, jetpTMin=1, jetRapidityCut=None, jetEtaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.clusterPower = -1
        self.jetRadius = jetRadius
        self.jetpTMin = jetpTMin
        self.jetRapidityCut = jetRapidityCut
        self.jetEtaCut = jetEtaCut

        if self.clusterPower == -1:
            self.jetDefinition = fj.JetDefinition(
                fj.antikt_algorithm, self.jetRadius)

        self.jetSelector = fj.SelectorPtMin(self.jetpTMin)
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
            fj_particles[i].set_user_index(int(hadrons[i].pid))

        return fj_particles


class pTYieldAnalysis(AnalysisBase):
    def __init__(self, pTBins=[], pTMin=0.1, rapidityCut=[-2, 2], etaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.pTBins = pTBins
        self.NpTBins = len(self.pTBins)-1

        self.pTMin = pTMin
        self.rapidityCut = rapidityCut
        self.etaCut = etaCut
        self.countStorage = [
            [0 for j in range(len(self.pTBins)-1)] for i in range(len(self.pThatBins)-1)]

    def analyzeEvent(self, particles):

        # filtering the particles first to save time
        particles = [p for p in particles if p.pid in self.ids and p.pT > self.pTMin and withinInterval(
            p.eta, self.etaCut) and withinInterval(p.y, self.rapidityCut)]
        for p in particles:
            i = findIndex(self.pTBins, p.pT)

            if i >= 0:
                self.countStorage[self.pThatIndex][i] += 1

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


class JetShapeAnalysis(JetAnalysisBase):
    def __init__(self, rBins, **kwargs):
        super().__init__(**kwargs)
        self.rBins = rBins
        self.NrBins = len(self.rBins)-1

        self.countStorage = [
            [0 for j in range(len(self.rBins)-1)] for i in range(len(self.pThatBins)-1)]

    def analyzeEvent(self, particles):

        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        for jet in jets_selected:
            constituents = jet.constituents()
            for hadron in constituents:
                dr = np.sqrt((hadron.eta()-jet.eta())**2 +
                             (hadron.phi()-jet.phi())**2)
                i = findIndex(self.rBins, dr)
                if i > 0:
                    self.countStorage[self.pThatIndex][i] += hadron.pt() / \
                        jet.pt()

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


class InclusiveJetpTYieldAnalysis(JetAnalysisBase, pTYieldAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analyzeEvent(self, particles):

        fjHadrons = self.fillFastJetConstituents(particles)

        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        for jet in jets_selected:
            i = findIndex(self.pTBins, jet.pt())
            if i >= 0:
                self.countStorage[self.pThatIndex][i] += 1


class HeavyJetpTYieldAnalysis(JetAnalysisBase, pTYieldAnalysis):
    def __init__(self, drCut=0.3, heavypTMin=5, heavyRapidityCut=None, heavyEtaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.drCut = 0.3
        self.heavypTMin = heavypTMin
        self.heavyRapidityCut = heavyRapidityCut
        self.heavyEtaCut = heavyEtaCut

    def analyzeEvent(self, particles):

        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        fjHadrons = [hadron for hadron in fjHadrons if hadron.user_index() in self.ids
                     and hadron.pt() > self.heavypTMin
                     and withinInterval(hadron.eta(), self.heavyEtaCut)
                     and withinInterval(hadron.rap(), self.heavyRapidityCut)]

        for jet in jets_selected:
            for hadron in fjHadrons:
                dr = np.sqrt((hadron.eta()-jet.eta())**2 +
                             (hadron.phi()-jet.phi())**2)
                if dr < self.drCut:
                    i = findIndex(self.pTBins, jet.pt())
                    if i >= 0:
                        self.countStorage[self.pThatIndex][i] += 1
                        break


class HeavyRadialProfileAnalysis(JetShapeAnalysis):
    def __init__(self, drCut=0.3, heavypTMin=5, heavyRapidityCut=None, heavyEtaCut=None, **kwargs):
        super().__init__(**kwargs)
        self.drCut = 0.3
        self.heavypTMin = heavypTMin
        self.heavyRapidityCut = heavyRapidityCut
        self.heavyEtaCut = heavyEtaCut

    def analyzeEvent(self, particles):

        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        for jet in jets_selected:
            constituents = jet.constituents()
            for hadron in constituents:
                dr = np.sqrt((hadron.eta()-jet.eta())**2 +
                             (hadron.phi()-jet.phi())**2)
                if hadron.user_index() in self.ids \
                        and dr < self.drCut \
                        and hadron.pt() > self.heavypTMin \
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

import numpy as np
import h5py
import random

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

class JetScapeReader:
    def __init__(self, fileName):
        self.fileName = fileName
        self.currentEventCount = 0
        self.currentCrossSection = 0
        self.currentHydroInfo = []
        self.hadronList = []

    def readEventHeader(self, strlist):
        self.currentCrossSection = float(strlist[6].strip())
        self.currentHydroInfo = [float(a.strip()) for a in strlist[8::2]]
        self.currentEventCount += 1

    def readAllEvents(self):
        with open(self.fileName) as f:
            for line in f:
                strlist = line.rstrip().split()
                if line.startswith("#"):   # A new event
                    if len(self.hadronList) == 0:
                        self.readEventHeader(strlist)
                        continue

                    yield self.hadronList
                    self.readEventHeader(strlist)
                    self.hadronList = []

                else:
                    i, pid, status, E, px, py, pz, eta, phi = [
                        float(a.strip()) for a in strlist
                    ]
                    self.hadronList.append([pid, E, px, py, pz, eta, phi])


class AnalysisBase:
    def __init__(self, pThatBins=[], pTBins=[], ids=[], rapidityCut=[-2, 2], outputFileName=""):
        self.outputFileName=outputFileName
        self.pThatBins = pThatBins
        self.pTBins = pTBins
        self.ids = ids
        self.rapidityCut = rapidityCut
        self.countStorage = [
            [0 for j in range(len(self.pTBins)-1)] for i in range(len(self.pThatBins)-1)]
        self.pThatEventCounts = [0 for i in range(len(self.pThatBins)-1)]
        self.pThatEventCrossSections = [
            0 for i in range(len(self.pThatBins)-1)]
        
    def setStatus(self, pThatIndex, reader):
        self.pThatIndex=pThatIndex
        self.pThatEventCrossSections[self.pThatIndex] = reader.currentCrossSection

    def addEvent(self, particles):
        self.pThatEventCounts[self.pThatIndex]+=1
        for particle in particles:
            pid, E, px, py, pz, eta, phi = particle
            if pid in self.ids and (eta >= self.rapidityCut[0] and eta <= self.rapidityCut[1]):
                pT = np.sqrt(px ** 2 + py ** 2)

                for i in range(len(self.pTBins) - 1):
                    if pT > self.pTBins[i] and pT < self.pTBins[i + 1]:
                        self.countStorage[self.pThatIndex][i] += 1

    def outputResult(self):
        if np.sum(self.pThatEventCounts)==0 : return
        rst = [0 for j in range(len(self.pTBins)-1)]
        err = [0 for j in range(len(self.pTBins)-1)]
        for pThat in range(len(self.pThatBins)-1):
            for pT in range(len(self.pTBins)-1):
                if self.pThatEventCounts[pThat] > 0:
                    normalizeFactor=self.pThatEventCounts[pThat]*(
                        self.pTBins[pT+1]-self.pTBins[pT])*(self.pTBins[pT+1]+self.pTBins[pT])/2*(self.rapidityCut[1]-self.rapidityCut[0])
                    rst[pT] += self.countStorage[pThat][pT]*self.pThatEventCrossSections[pThat]/normalizeFactor
                    err[pT] += self.countStorage[pThat][pT]*self.pThatEventCrossSections[pThat]**2/normalizeFactor**2

        ptBinsAvg = (np.array(self.pTBins[0:-1])+np.array(self.pTBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose([ptBinsAvg, rst, err]))


class FlowAnalysis(AnalysisBase):
    def __init__(self, **kwargs):
        super(FlowAnalysis, self).__init__(**kwargs)
        self.allHydros=[]
        self.Q2_Re = [[[] for j in range(
            len(self.pTBins)-1)] for i in range(len(self.pThatBins)-1)]
        self.Q2_Im = [[[] for j in range(
            len(self.pTBins)-1)] for i in range(len(self.pThatBins)-1)]
        self.Q0 = [[[] for j in range(
            len(self.pTBins)-1)] for i in range(len(self.pThatBins)-1)]

    def setStatus(self, pThatIndex, reader):
        self.pThatIndex=pThatIndex
        self.pThatEventCrossSections[self.pThatIndex] = reader.currentCrossSection
        self.currentHydro=reader.currentHydroInfo
        if self.currentHydro not in self.allHydros:
            #print("hydro increasd to "+ str(len(self.allHydros)))
            #print(hydro)
            self.allHydros.append(self.currentHydro)
            for pThat in range(len(self.pThatBins)-1):
                for pT in range(len(self.pTBins)-1):
                    self.Q2_Re[pThat][pT].append(0)
                    self.Q2_Im[pThat][pT].append(0)
                    self.Q0[pThat][pT].append(0)

    def addEvent(self,particles):
        self.pThatEventCounts[self.pThatIndex]+=1
        hydroId=self.allHydros.index(self.currentHydro)
        
        for particle in particles:
            pid, E, px, py, pz, eta, phi = particle
            if pid in self.ids and (eta >= self.rapidityCut[0] and eta <= self.rapidityCut[1]):
                pT = np.sqrt(px ** 2 + py ** 2)

                for i in range(len(self.pTBins) - 1):
                    if pT > self.pTBins[i] and pT < self.pTBins[i + 1]:
                        self.Q2_Re[self.pThatIndex][i][hydroId] += np.cos(2 * phi)
                        self.Q2_Im[self.pThatIndex][i][hydroId] += np.sin(2 * phi)
                        self.Q0[self.pThatIndex][i][hydroId] += 1

    def outputResult(self):
        if np.sum(self.pThatEventCounts)==0 : return
        v2_ch_rms = np.sqrt(np.mean(np.array(self.allHydros)[:, 3] ** 2))
        v2_all = [[] for j in range(len(self.pTBins)-1)]
        rst = [0 for j in range(len(self.pTBins)-1)]
        for hydroId in range(len(self.allHydros)):
            for pT in range(len(self.pTBins)-1):
                temp_Q2_Re = 0
                temp_Q2_Im = 0
                temp_Q0 = 0
                for pThat in range(len(self.pThatBins)-1):
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
                psi_temp= 0.5 * \
                    np.arctan2(temp_Q2_Im, temp_Q2_Re)
                weight = temp_Q0*self.allHydros[hydroId][2]
                v2_ch = self.allHydros[hydroId][3]
                psi_ch = self.allHydros[hydroId][4]
    
                v2_all[pT].append(
                    [v2_temp * v2_ch * np.cos(2 * (psi_temp - psi_ch)), weight])

        for i in range(len(self.pTBins)-1):
            v2 = np.array(v2_all[i])
            if len(v2) > 0:
                rst[i] = (weighted_avg_and_std(
                    v2[:, 0], v2[:, 1])[0]/v2_ch_rms)

        ptBinsAvg = (np.array(self.pTBins[0:-1])+np.array(self.pTBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose([ptBinsAvg, rst]))

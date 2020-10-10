import numpy as np
import h5py
import random
import fastjet as fj
import fjext


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


def withinInterval(y, interval):
    if interval == None:
        return True
    if len(interval) == 2:
        assert(interval[0] < interval[1])
        if y >= interval[0] and y < interval[1]:
            return True
    return False


class Particle:
    def __init__(self, pid, status, E, px, py, pz):
        self.pid = pid
        self.status = status
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz
        self.pT = np.sqrt(px**2+py**2)
        self.y = 0.5 * np.log((abs(E + pz) + 0.0000001) /
                              (abs(E - pz) + 0.0000001))
        modP = np.sqrt(px**2+py**2+pz**2)
        self.eta = 0.5 * \
            np.log((abs(modP + pz) + 0.0000001) / (abs(modP - pz) + 0.0000001))
        self.phi = np.arctan2(px, py)


class JetScapeReader:
    def __init__(self, fileName):
        self.fileName = fileName
        self.currentEventCount = 0
        self.currentCrossSection = 0
        self.currentHydroInfo = []
        self.particleList = []

    def readEventHeader(self, strlist):
        '''
        currently only support a single header format
        with event cross section and hydro flow information
        '''
        assert(len(strlist) == 25)

        self.currentCrossSection = float(strlist[6].strip())
        self.currentHydroInfo = [float(a.strip()) for a in strlist[8::2]]
        self.currentEventCount += 1

    def readAllEvents(self):
        with open(self.fileName) as f:
            for line in f:
                strlist = line.rstrip().split()
                if line.startswith("#"):   # A new event
                    if len(self.particleList) == 0:
                        self.readEventHeader(strlist)
                        continue

                    yield self.particleList
                    self.readEventHeader(strlist)
                    self.particleList = []

                else:
                    if len(strlist) == 9:
                        i, pid, status, E, px, py, pz, eta, phi = [
                            float(a.strip()) for a in strlist
                        ]
                        self.particleList.append(
                            Particle(pid, status, E, px, py, pz)
                        )
                    elif len(strlist) == 6:
                        pid, status, E, px, py, pz = [
                            float(a.strip()) for a in strlist
                        ]
                        self.particleList.append(
                            Particle(pid, status, E, px, py, pz))
                    else:
                        print("Unknown particle length")


'''
Assumption: 
1.the simulation is setup with many pThatBins and we want to combine them in the analysis
2. the simulation is interested in particles pf certain id and status
'''


class AnalysisBase:
    def __init__(self, pThatBins=[], ids=[], status=[],  outputFileName=""):
        self.outputFileName = outputFileName

        self.ids = ids
        self.status = status

        self.pThatBins = pThatBins
        self.NpThatBins = len(self.pThatBins)-1

        self.pThatEventCounts = [0 for i in range(self.NpThatBins)]
        self.pThatEventCrossSections = [
            0 for i in range(self.NpThatBins)]

    def setCrossSection(self, pThatIndex, CS):
        self.pThatIndex = pThatIndex
        self.pThatEventCrossSections[self.pThatIndex]=CS

    def setStatus(self, pThatIndex, reader):
        self.pThatIndex = pThatIndex

    def analyzeEvent(self, particles):
        raise NotImplementedError()

    def outputResult(self):
        raise NotImplementedError()


class JetAnalysisBase(AnalysisBase):
    def __init__(self, jetRadius=0.4, jetpTMin=1, jetRapidityCut=[-2, 2], jetEtaCut=None, **kwargs):
        super(JetAnalysisBase, self).__init__(**kwargs)
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

        px = [hadron.px for hadron in hadrons]
        py = [hadron.py for hadron in hadrons]
        pz = [hadron.pz for hadron in hadrons]
        e = [hadron.E for hadron in hadrons]

        # Create a vector of fastjet::PseudoJets from arrays of px,py,pz,e
        fj_particles = fjext.vectorize_px_py_pz_e(px, py, pz, e)

        return fj_particles


class pTYieldAnalysis(AnalysisBase):
    def __init__(self, pTBins=[], rapidityCut=[-2, 2], etaCut=None, **kwargs):
        super(pTYieldAnalysis, self).__init__(**kwargs)
        self.pTBins = pTBins
        self.NpTBins = len(self.pTBins)-1

        self.rapidityCut = rapidityCut
        self.etaCut = etaCut
        self.countStorage = [
            [0 for j in range(len(self.pTBins)-1)] for i in range(len(self.pThatBins)-1)]

    def setStatus(self, pThatIndex, reader):
        self.pThatIndex = pThatIndex
        self.pThatEventCrossSections[self.pThatIndex] = reader.currentCrossSection

    def analyzeEvent(self, particles):
        self.pThatEventCounts[self.pThatIndex] += 1
        for p in particles:
            if p.pid in self.ids and withinInterval(p.eta, self.etaCut) and withinInterval(p.y, self.rapidityCut):

                for i in range(self.NpTBins):
                    if p.pT > self.pTBins[i] and p.pT < self.pTBins[i + 1]:
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
                        self.pTBins[pT+1]-self.pTBins[pT])*(self.pTBins[pT+1]+self.pTBins[pT])/2*(self.rapidityCut[1]-self.rapidityCut[0])
                    rst[pT] += self.countStorage[pThat][pT] * \
                        self.pThatEventCrossSections[pThat]/normalizeFactor
                    err[pT] += self.countStorage[pThat][pT] * \
                        self.pThatEventCrossSections[pThat]**2 / \
                        normalizeFactor**2

        err = [np.sqrt(x) for x in err]
        ptBinsAvg = (np.array(self.pTBins[0:-1])+np.array(self.pTBins[1:]))/2
        np.savetxt(self.outputFileName, np.transpose([ptBinsAvg, rst, err]))


class InclusiveJetpTYieldAnalysis(JetAnalysisBase, pTYieldAnalysis):
    def __init__(self, **kwargs):
        super(InclusiveJetpTYieldAnalysis, self).__init__(**kwargs)

    def analyzeEvent(self, particles):
        self.pThatEventCounts[self.pThatIndex] += 1
        fjHadrons = self.fillFastJetConstituents(particles)
        cs = fj.ClusterSequence(fjHadrons, self.jetDefinition)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jetSelector(jets)

        for jet in jets_selected:
            for i in range(self.NpTBins):
                if jet.pt() > self.pTBins[i] and jet.pt() < self.pTBins[i + 1]:
                    self.countStorage[self.pThatIndex][i] += 1


class FlowAnalysis(pTYieldAnalysis):
    def __init__(self, **kwargs):
        super(FlowAnalysis, self).__init__(**kwargs)

        self.allHydros = []
        self.Q2_Re = [[[] for j in range(
            self.NpTBins)] for i in range(self.NpThatBins)]
        self.Q2_Im = [[[] for j in range(
            self.NpTBins)] for i in range(self.NpThatBins)]
        self.Q0 = [[[] for j in range(
            self.NpTBins)] for i in range(self.NpThatBins)]

    def setStatus(self, pThatIndex, reader):
        self.pThatIndex = pThatIndex
        self.pThatEventCrossSections[self.pThatIndex] = reader.currentCrossSection
        self.currentHydro = reader.currentHydroInfo
        if self.currentHydro not in self.allHydros:

            self.allHydros.append(self.currentHydro)
            for pThat in range(self.NpThatBins):
                for pT in range(self.NpTBins):
                    self.Q2_Re[pThat][pT].append(0)
                    self.Q2_Im[pThat][pT].append(0)
                    self.Q0[pThat][pT].append(0)

    def analyzeEvent(self, particles):
        self.pThatEventCounts[self.pThatIndex] += 1
        hydroId = self.allHydros.index(self.currentHydro)

        for p in particles:
            if p.pid in self.ids and withinInterval(p.eta, self.etaCut) and withinInterval(p.y, self.rapidityCut):
                for i in range(self.NpTBins):
                    if p.pT > self.pTBins[i] and p.pT < self.pTBins[i + 1]:
                        self.Q2_Re[self.pThatIndex][i][hydroId] += np.cos(
                            2 * p.phi)
                        self.Q2_Im[self.pThatIndex][i][hydroId] += np.sin(
                            2 * p.phi)
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
        np.savetxt(self.outputFileName, np.transpose([ptBinsAvg, rst]))

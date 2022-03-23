#include "JetAnalyzer.h"

class JetShapeAnalyzer : public JetAnalyzer
{
public:
	JetShapeAnalyzer(vector<double> outputBins, double clusterPower, double jetRadius, double jetpTMin, double jetyMin, double jetyMax) : JetAnalyzer("JetShape", outputBins, clusterPower, jetRadius, jetpTMin, jetyMin, jetyMax){};

	void doJetFinding(std::vector<fastjet::PseudoJet> fjInputs, double weight)
	{
		//fastjet.init();
		fastjet::JetDefinition jetDef;
		if (clusterPower == 1 || clusterPower == -1)
		{
			jetDef = fastjet::JetDefinition(fastjet::genkt_algorithm, jetRadius, clusterPower);
		}
		else
		{
			jetDef = fastjet::JetDefinition(fastjet::cambridge_algorithm, jetRadius);
		}
		fastjet::Selector select_rapidity = fastjet::SelectorRapRange(jetyMin, jetyMax);

		vector<fastjet::PseudoJet> inclusiveJets, jets;
		fastjet::ClusterSequence clustSeq(fjInputs, jetDef);
		inclusiveJets = clustSeq.inclusive_jets(jetpTMin);
		std::vector<fastjet::PseudoJet> selected_jets = select_rapidity(inclusiveJets);
		jets = sorted_by_pt(selected_jets);

		std::vector<int> jet_ct = std::vector<int>(outputBins.size() - 1, 0);

		for (int k = 0; k < jets.size(); k++)
		{
			double jeteta = jets[k].eta();
			double jetphi = jets[k].phi();
			vector<fastjet::PseudoJet> constituents = jets[k].constituents();
			for (int p = 0; p < constituents.size(); p++)
			{
				//cout<<constituents[p].user_info<LidoParticleInfo>().T0 <<endl;
				if (constituents[p].pt() > 1)
				{
					double eta = constituents[p].eta();
					double phi = constituents[p].phi();
					double dr = sqrt(pow((eta - jeteta), 2) + pow((phi - jetphi), 2));
					for (unsigned int j = 0; j < outputBins.size() - 1; j++)
					{
						if (dr >= outputBins[j] && dr < outputBins[j + 1])
						{
							jet_ct[j] += constituents[p].pt() / jets[k].pt();
							break;
						}
					}
				}
			}
		}

		for (unsigned int j = 0; j < outputBins.size() - 1; j++)
		{
			rst[j] += jet_ct[j] * weight;
			sqSum[j] += jet_ct[j] * pow(weight, 2);
		}
	}

};
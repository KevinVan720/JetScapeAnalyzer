#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <exception>
#include <random>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <sstream>
#include <unistd.h>

#include "DJetShapeAnalyzer.h"

int main(int argc, char *argv[])
{
    std::string eventFile(argv[1]);
    double weight=atof(argv[2]);

    std::vector<double> shaperbins({0, 0.05, 0.1, 0.3, 0.5});

    std::vector<JetAnalyzer *> analyzers;

    analyzers.push_back(new DJetShapeAnalyzer(shaperbins, -1, 0.3, 60, 0, 1.6));

    std::vector<fastjet::PseudoJet> fjInputs;
    std::ifstream infile(eventFile);
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int id;
        int pid;
        int pstat;
        double px;
        double py;
        double pz;
        double e;
        double eta;
        double phi;
        std::string str;

        if (line[0] == '#')
        {
            //eventID+=1;
            for (int i = 0; i < analyzers.size(); i++)
            {
                analyzers[i]->doJetFinding(fjInputs, weight* pow(10, 9));
            }
            fjInputs.clear();
        }
        else
        {
            iss >> id >> pid >> pstat >> e >> px >> py >> pz >> eta >> phi;
            fjInputs.push_back(fastjet::PseudoJet(px, py, pz, e));
            fjInputs[fjInputs.size() - 1].set_user_index(pid);
        }
    }
    infile.close();

    for (int i = 0; i < analyzers.size(); i++)
    {
        std::string name=argv[1];
        analyzers[i]->outputResults(name.substr(0,name.find_last_of('.')));
    }

    return 0;
}

# deal with pythia
PYTHIADIR   = /usr/local
PYTHIAFLAGS = -I$(PYTHIADIR)/include -O2 -pedantic -W -Wall -Wshadow -fPIC
PYTHIALIBS  =  -L$(PYTHIADIR)/lib -Wl,-rpath,$(PYTHIADIR) -lpythia8 -ldl

# deal with fastjet
FASTJETDIR  = /usr/local
FASTJETARGS = `$(FASTJETDIR)/bin/fastjet-config --cxxflags --libs`

# LHAPDF if you want to replace Pythia's dummy default
LHAPDFARGS =
#LHAPDFARGS = -L$(HOME)/local/lib -lLHAPDF -I$(HOME)/local/include

#main72:
#	g++ -o main72 main72.cc $(PYTHIAFLAGS) $(PYTHIALIBS) $(FASTJETARGS) $(LHAPDFARGS) 
#main71:
#	g++ -o main71 main71.cc $(PYTHIAFLAGS) $(PYTHIALIBS) $(FASTJETARGS) $(LHAPDFARGS) 
#jet:
#	g++ -o Jet Jet_cs.cpp $(PYTHIAFLAGS) $(PYTHIALIBS) $(FASTJETARGS) $(LHAPDFARGS) 
jet_simu:
	g++ -std=c++11 -o 2Dhydro-jet-analysis 2Dhydro-jet-analysis.cpp $(PYTHIAFLAGS) $(PYTHIALIBS) $(FASTJETARGS) $(LHAPDFARGS) 

// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "Dataset.h"
#include "mpi-extra.h"
#include "utility.h"
#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    bool structureInfo = false;
    int numProcs = 0;
    int myRank = 0;

    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <info>\n"
             << "       <info> ... Write structure information for debugging "
                "to \"structure.out (0/1)\".\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n"
             << "       - \"weights.%%03d.data\" (weights files)\n";
        return 1;
    }

    structureInfo = (bool)atoi(argv[1]);
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    ofstream logFile;
    logFile.open("nnp-predict.log");
    Dataset dataset;
    if (myRank != 0) dataset.log.writeToStdout = false;
    dataset.log.registerStreamPointer(&logFile);
    dataset.setupMPI();
    dataset.initialize();
    dataset.loadSettingsFile();
    dataset.setupGeneric();
    dataset.setupSymmetryFunctionScaling();
    dataset.setupSymmetryFunctionStatistics(false, false, true, false);
    dataset.setupNeuralNetworkWeights();
    dataset.log << "\n";
    dataset.log << "*** PREDICTION **************************"
                   "**************************************\n";
    dataset.log << "\n";
    dataset.log << "Reading structure file...\n";
    dataset.distributeStructures(false); // false = don't shuffle
    if (dataset.useNormalization()) dataset.toNormalizedUnits();
    dataset.log << strpr("Found %zu configurations in input.data\n", dataset.structures.size());
    dataset.log << "Calculating NNP predictions...\n";
    
    // Process each structure
    for (size_t i = 0; i < dataset.structures.size(); ++i)
    {
        Structure& s = dataset.structures[i];
        
        // Calculate neighbor list and predictions for this structure
        s.calculateNeighborList(dataset.getMaxCutoffRadius());
        dataset.evaluateNNP(s);
        
        if (dataset.useNormalization()) 
        {
            s.toPhysicalUnits(dataset.getMeanEnergy(), 
                             dataset.getConvEnergy(), 
                             dataset.getConvLength(), 
                             dataset.getConvCharge());
        }
        dataset.addEnergyOffset(s, false);
        dataset.addEnergyOffset(s, true);
        
        dataset.log << strpr("Configuration %zu: %zu atoms\n", i + 1, s.numAtoms);
        dataset.log << strpr("  NNP total energy: %16.8E\n", s.energy);
        
        if (dataset.getNnpType() == Mode::NNPType::HDNNP_4G)
        {
            dataset.log << strpr("  NNP electrostatic energy: %16.8E\n", s.energyElec);
            dataset.log << strpr("  NNP total charge: %16.8E (ref: %16.8E)\n", s.charge, s.chargeRef);
        }
    }
    
    dataset.log << "\n";
    dataset.logEwaldCutoffs();
    dataset.log << "-----------------------------------------"
                   "--------------------------------------\n";
    dataset.log << "Writing output files...\n";
    dataset.log << " - energy.out\n";
    ofstream file;
    file.open("energy.out");

    // File header.
    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Energy comparison.");
    colSize.push_back(10);
    colName.push_back("conf");
    colInfo.push_back("Configuration index (starting with 1).");
    colSize.push_back(10);
    colName.push_back("natoms");
    colInfo.push_back("Number of atoms in configuration.");
    colSize.push_back(24);
    colName.push_back("Eref");
    colInfo.push_back("Reference potential energy.");
    colSize.push_back(24);
    colName.push_back("Ennp");
    colInfo.push_back("Potential energy predicted by NNP.");
    colSize.push_back(24);
    colName.push_back("Ediff");
    colInfo.push_back("Difference in energy per atom between reference and "
                      "NNP prediction.");
    colSize.push_back(24);
    colName.push_back("E_offset");
    colInfo.push_back("Sum of atomic offset energies "
                      "(included in column Ennp).");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    // Write energy data for all structures
    for (size_t i = 0; i < dataset.structures.size(); ++i)
    {
        Structure& s = dataset.structures[i];
        file << strpr("%10zu %10zu %24.16E %24.16E %24.16E %24.16E\n",
                      s.index + 1,
                      s.numAtoms,
                      s.energyRef,
                      s.energy,
                      (s.energyRef - s.energy) / s.numAtoms,
                      dataset.getEnergyOffset(s));
    }
    file.close();

    dataset.log << " - nnatoms.out\n";
    file.open("nnatoms.out");

    // File header.
    title.clear();
    colName.clear();
    colInfo.clear();
    colSize.clear();
    title.push_back("Energy contributions calculated from NNP.");
    colSize.push_back(10);
    colName.push_back("conf");
    colInfo.push_back("Configuration index (starting with 1).");
    colSize.push_back(10);
    colName.push_back("index");
    colInfo.push_back("Atom index (starting with 1).");
    colSize.push_back(3);
    colName.push_back("Z");
    colInfo.push_back("Nuclear charge of atom.");
    colSize.push_back(24);
    colName.push_back("Qref");
    colInfo.push_back("Reference atomic charge.");
    colSize.push_back(24);
    colName.push_back("Qnnp");
    colInfo.push_back("NNP atomic charge.");
    colSize.push_back(24);
    colName.push_back("Eref_atom");
    colInfo.push_back("Reference atomic energy contribution.");
    colSize.push_back(24);
    colName.push_back("Ennp_atom");
    colInfo.push_back("Atomic energy contribution (physical units, no mean "
                      "or offset energy added).");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    // Write atomic data for all structures
    for (size_t i = 0; i < dataset.structures.size(); ++i)
    {
        Structure& s = dataset.structures[i];
        for (vector<Atom>::const_iterator it = s.atoms.begin();
             it != s.atoms.end(); ++it)
        {
            file << strpr("%10zu %10zu %3zu %24.16E %24.16E %24.16E %24.16E\n",
                          s.index + 1,
                          it->index + 1,
                          dataset.elementMap.atomicNumber(it->element),
                          it->chargeRef,
                          it->charge,
                          0.0,
                          it->energy);
        }
    }
    file.close();

    dataset.log << " - nnforces.out\n";
    file.open("nnforces.out");

    // File header.
    title.clear();
    colName.clear();
    colInfo.clear();
    colSize.clear();
    title.push_back("Atomic force comparison (ordered by atom index).");
    colSize.push_back(10);
    colName.push_back("conf");
    colInfo.push_back("Configuration index (starting with 1).");
    colSize.push_back(10);
    colName.push_back("index");
    colInfo.push_back("Atom index (starting with 1).");
    colSize.push_back(24);
    colName.push_back("fxRef");
    colInfo.push_back("Reference force in x direction.");
    colSize.push_back(24);
    colName.push_back("fyRef");
    colInfo.push_back("Reference force in y direction.");
    colSize.push_back(24);
    colName.push_back("fzRef");
    colInfo.push_back("Reference force in z direction.");
    colSize.push_back(24);
    colName.push_back("fx");
    colInfo.push_back("Force in x direction.");
    colSize.push_back(24);
    colName.push_back("fy");
    colInfo.push_back("Force in y direction.");
    colSize.push_back(24);
    colName.push_back("fz");
    colInfo.push_back("Force in z direction.");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    // Write force data for all structures
    for (size_t i = 0; i < dataset.structures.size(); ++i)
    {
        Structure& s = dataset.structures[i];
        for (vector<Atom>::const_iterator it = s.atoms.begin();
             it != s.atoms.end(); ++it)
        {
            file << strpr("%10zu %10zu %24.16E %24.16E %24.16E %24.16E %24.16E "
                          "%24.16E\n",
                          s.index + 1,
                          it->index + 1,
                          it->fRef[0],
                          it->fRef[1],
                          it->fRef[2],
                          it->f[0],
                          it->f[1],
                          it->f[2]);
        }
    }
    file.close();

    dataset.log << "Writing structures with NNP predictions "
                   "to \"output.data\".\n";
    file.open("output.data");
    for (size_t i = 0; i < dataset.structures.size(); ++i)
    {
        dataset.structures[i].writeToFile(&file, false);
    }
    file.close();

    if (structureInfo)
    {
        dataset.log << "Writing detailed structure information to "
                       "\"structure.out\".\n";
        file.open("structure.out");
        for (size_t i = 0; i < dataset.structures.size(); ++i)
        {
            file << "Configuration " << (i + 1) << ":\n";
            vector<string> info = dataset.structures[i].info();
            appendLinesToFile(file, info);
            file << "\n";
        }
        file.close();
    }

    dataset.log << "Finished.\n";
    dataset.log << "*****************************************"
                   "**************************************\n";
    logFile.close();

    MPI_Finalize();
    return 0;
}

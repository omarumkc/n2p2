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

#include "Atom.h"
#include "utility.h"
#include <cinttypes> // PRId64
#include <cmath>     // fabs
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::range_error

using namespace std;
using namespace nnp;

Atom::Atom() : hasNeighborList               (false),
               NeighborListIsSorted          (false),
               hasSymmetryFunctions          (false),
               hasSymmetryFunctionDerivatives(false),
               useChargeNeuron               (false),
               index                         (0    ),
               indexStructure                (0    ),
               tag                           (0    ),
               element                       (0    ),
               numNeighbors                  (0    ),
               numNeighborsUnique            (0    ),
               numSymmetryFunctions          (0    ),
               energy                        (0.0  ),
               dEelecdQ                      (0.0  ),
               chi                           (0.0  ),
               charge                        (0.0  ),
               chargeRef                     (0.0  )
{
}

#ifdef N2P2_FULL_SFD_MEMORY
void Atom::collectDGdxia(size_t indexAtom,
                         size_t indexComponent,
                         double maxCutoffRadius)
{
    for (size_t i = 0; i < dGdxia.size(); i++)
    {
        dGdxia[i] = 0.0;
    }
    for (size_t i = 0; i < getStoredMinNumNeighbors(maxCutoffRadius); i++)
    {
        if (neighbors[i].index == indexAtom)
        {
            for (size_t j = 0; j < numSymmetryFunctions; ++j)
            {
                dGdxia[j] += neighbors[i].dGdr[j][indexComponent];
            }
        }
    }
    if (index == indexAtom)
    {
        for (size_t i = 0; i < numSymmetryFunctions; ++i)
        {
            dGdxia[i] += dGdr[i][indexComponent];
        }
    }

    return;
}
#endif

void Atom::toNormalizedUnits(double convEnergy,
                             double convLength,
                             double convCharge)
{
    energy *= convEnergy;
    r *= convLength;
    f *= convEnergy / convLength;
    fRef *= convEnergy / convLength;
    charge *= convCharge;
    chargeRef *= convCharge;
    if (hasSymmetryFunctionDerivatives)
    {
        for (size_t i = 0; i < numSymmetryFunctions; ++i)
        {
            dEdG.at(i) *= convEnergy;
            dGdr.at(i) /= convLength;
#ifdef N2P2_FULL_SFD_MEMORY
            dGdxia.at(i) /= convLength;
#endif
        }
        // Take care of extra charge neuron.
        if (useChargeNeuron)
            dEdG.at(numSymmetryFunctions) *= convEnergy / convCharge;
    }

    if (hasNeighborList)
    {
        for (vector<Neighbor>::iterator it = neighbors.begin();
             it != neighbors.end(); ++it)
        {
            it->d *= convLength;
            it->dr *= convLength;
            if (hasSymmetryFunctionDerivatives)
            {
                for (size_t i = 0; i < dGdr.size(); ++i)
                {
                    dGdr.at(i) /= convLength;
                }
            }
        }
    }

    return;
}

void Atom::toPhysicalUnits(double convEnergy,
                           double convLength,
                           double convCharge)
{
    energy /= convEnergy;
    r /= convLength;
    f /= convEnergy / convLength;
    fRef /= convEnergy / convLength;
    charge /= convCharge;
    chargeRef /= convCharge;
    if (hasSymmetryFunctionDerivatives)
    {
        for (size_t i = 0; i < numSymmetryFunctions; ++i)
        {
            dEdG.at(i) /= convEnergy;
            dGdr.at(i) *= convLength;
#ifdef N2P2_FULL_SFD_MEMORY
            dGdxia.at(i) *= convLength;
#endif
        }
        // Take care of extra charge neuron.
        if (useChargeNeuron)
            dEdG.at(numSymmetryFunctions) /= convEnergy / convCharge;
    }

    if (hasNeighborList)
    {
        for (vector<Neighbor>::iterator it = neighbors.begin();
             it != neighbors.end(); ++it)
        {
            it->d /= convLength;
            it->dr /= convLength;
            if (hasSymmetryFunctionDerivatives)
            {
                for (size_t i = 0; i < dGdr.size(); ++i)
                {
                    dGdr.at(i) *= convLength;
                }
            }
        }
    }

    return;
}

void Atom::allocate(bool all, double const maxCutoffRadius)
{
    if (numSymmetryFunctions == 0)
    {
        throw range_error("ERROR: Number of symmetry functions set to"
                          "zero, cannot allocate.\n");
    }

    // Clear all symmetry function related vectors (also for derivatives).
    G.clear();
    dEdG.clear();
    dQdG.clear();
    dChidG.clear();
#ifdef N2P2_FULL_SFD_MEMORY
    dGdxia.clear();
#endif
    dGdr.clear();

    // Don't need any allocation of neighbors outside the symmetry function
    // cutoff.
    size_t const numSymFuncNeighbors = getStoredMinNumNeighbors(maxCutoffRadius);
    for (size_t i = 0; i < numSymFuncNeighbors; ++i)
    {
        Neighbor& n = neighbors.at(i);
#ifndef N2P2_NO_SF_CACHE
        n.cache.clear();
#endif
        n.dGdr.clear();
    }

    // Reset status of symmetry functions and derivatives.
    hasSymmetryFunctions           = false;
    hasSymmetryFunctionDerivatives = false;

    // Resize vectors (derivatives only if requested).
    G.resize(numSymmetryFunctions, 0.0);
    if (all)
    {
#ifndef N2P2_FULL_SFD_MEMORY
        if (numSymmetryFunctionDerivatives.size() == 0)
        {
            throw range_error("ERROR: Number of symmetry function derivatives"
                              " unset, cannot allocate.\n");
        }
#endif
        if (useChargeNeuron) dEdG.resize(numSymmetryFunctions + 1, 0.0);
        else                 dEdG.resize(numSymmetryFunctions, 0.0);
        dQdG.resize(numSymmetryFunctions, 0.0);
        dChidG.resize(numSymmetryFunctions,0.0);
#ifdef N2P2_FULL_SFD_MEMORY
        dGdxia.resize(numSymmetryFunctions, 0.0);
#endif
        dGdr.resize(numSymmetryFunctions);
    }
    for (size_t i = 0; i < numSymFuncNeighbors; ++i)
    {
        Neighbor& n = neighbors.at(i);
#ifndef N2P2_NO_SF_CACHE
        n.cache.resize(cacheSizePerElement.at(n.element),
                         -numeric_limits<double>::max());
#endif
        if (all)
        {
#ifndef N2P2_FULL_SFD_MEMORY
            n.dGdr.resize(numSymmetryFunctionDerivatives.at(n.element));
#else
            n.dGdr.resize(numSymmetryFunctions);
#endif
        }
    }

    return;
}

void Atom::free(bool all, double const maxCutoffRadius)
{
    size_t const numSymFuncNeighbors = getStoredMinNumNeighbors(maxCutoffRadius);
    if (all)
    {
        G.clear();
        vector<double>(G).swap(G);
        hasSymmetryFunctions = false;

#ifndef N2P2_NO_SF_CACHE
        for (size_t i = 0; i < numSymFuncNeighbors; ++i)
        {
            Neighbor& n = neighbors.at(i);
            n.cache.clear();
            vector<double>(n.cache).swap(n.cache);
        }
#endif
    }

    dEdG.clear();
    vector<double>(dEdG).swap(dEdG);
    dQdG.clear();
    vector<double>(dQdG).swap(dQdG);
    dChidG.clear();
    vector<double>(dChidG).swap(dChidG);
#ifdef N2P2_FULL_SFD_MEMORY
    dGdxia.clear();
    vector<double>(dGdxia).swap(dGdxia);
#endif
    dGdr.clear();
    vector<Vec3D>(dGdr).swap(dGdr);

    for (size_t i = 0; i < numSymFuncNeighbors; ++i)
    {
        Neighbor& n = neighbors.at(i);
        n.dGdr.clear();
        vector<Vec3D>(n.dGdr).swap(n.dGdr);
    }
    hasSymmetryFunctionDerivatives = false;

    return;
}

void Atom::clearNeighborList()
{
    clearNeighborList(numNeighborsPerElement.size());

    return;
}

void Atom::clearNeighborList(size_t const numElements)
{
    free(true);
    numNeighbors = 0;
    numNeighborsPerElement.resize(numElements, 0);
    numNeighborsUnique = 0;
    neighborsUnique.clear();
    vector<size_t>(neighborsUnique).swap(neighborsUnique);
    neighbors.clear();
    vector<Atom::Neighbor>(neighbors).swap(neighbors);
    hasNeighborList = false;

    return;
}

size_t Atom::calculateNumNeighbors(double const cutoffRadius) const
{
    // neighborCutoffs map is only constructed when the neighbor list is sorted.
    if (unorderedMapContainsKey(neighborCutoffs, cutoffRadius))
        return neighborCutoffs.at(cutoffRadius);
    else
    {
        size_t numNeighborsLocal = 0;

        for (vector<Neighbor>::const_iterator it = neighbors.begin();
             it != neighbors.end(); ++it)
        {
            if (it->d <= cutoffRadius)
            {
                numNeighborsLocal++;
            }
        }

        return numNeighborsLocal;
    }
}

size_t Atom::getStoredMinNumNeighbors(double const cutoffRadius) const
{
     // neighborCutoffs map is only constructed when the neighbor list is sorted.
    if (unorderedMapContainsKey(neighborCutoffs, cutoffRadius))
        return neighborCutoffs.at(cutoffRadius);
    else
    {
        return numNeighbors;
    }
}

bool Atom::isNeighbor(size_t index) const
{
    for (auto const& n : neighbors) if (n.index == index) return true;

    return false;
}

void Atom::updateError(string const&        property,
                       map<string, double>& error,
                       size_t&              count) const
{
    if (property == "force")
    {
        count += 3;
        error.at("RMSE") += (fRef - f).norm2();
        error.at("MAE") += (fRef - f).l1norm();
    }
    else if (property == "charge")
    {
        count++;
        error.at("RMSE") += (chargeRef - charge) * (chargeRef - charge);
        error.at("MAE") += fabs(chargeRef - charge);
    }
    else
    {
        throw runtime_error("ERROR: Unknown property for error update.\n");
    }

    return;
}

Vec3D Atom::calculateSelfForceShort() const
{
    Vec3D selfForce{};
    for (size_t i = 0; i < numSymmetryFunctions; ++i)
    {
        selfForce -= dEdG.at(i) * dGdr.at(i);
    }
    return selfForce;
}

Vec3D Atom::calculatePairForceShort(Neighbor const&             neighbor,
                                    vector<vector<size_t> >
                                    const *const                tableFull) const
{
    Vec3D pairForce{};
#ifndef N2P2_FULL_SFD_MEMORY
    if (!tableFull) throw runtime_error(
                "ERROR: tableFull must not be null pointer");
    vector<size_t> const& table = tableFull->at(neighbor.element);
    for (size_t i = 0; i < neighbor.dGdr.size(); ++i)
    {
        pairForce -= dEdG.at(table.at(i)) * neighbor.dGdr.at(i);
    }
#else
    for (size_t i = 0; i < numSymmetryFunctions; ++i)
    {
        pairForce -= dEdG.at(i) * neighbor.dGdr.at(i);
    }
#endif
    return pairForce;
}

Vec3D Atom::calculateDChidr(size_t const atomIndexOfR,
                            double const maxCutoffRadius,
                            vector<vector<size_t> > const *const tableFull) const
{
#ifndef N2P2_FULL_SFD_MEMORY
    if (!tableFull) throw runtime_error(
                "ERROR: tableFull must not be null pointer");
#endif
    Vec3D dChidr{};
    // need to add this case because the loop over the neighbors
    // does not include the contribution dChi_i/dr_i.
    if (atomIndexOfR == index)
    {
        for (size_t k = 0; k < numSymmetryFunctions; ++k)
        {
            dChidr += dChidG.at(k) * dGdr.at(k);
        }
    }

    size_t numNeighbors = getStoredMinNumNeighbors(maxCutoffRadius);
    for (size_t m = 0; m < numNeighbors; ++m)
    {
        Atom::Neighbor const& n = neighbors.at(m);
	// atomIndexOfR must not be larger than ~max(int64_t)/2.
        if (n.tag == (int64_t)atomIndexOfR)
        {
#ifndef N2P2_FULL_SFD_MEMORY
            vector<size_t> const& table = tableFull->at(n.element);
            for (size_t k = 0; k < n.dGdr.size(); ++k)
            {
                dChidr += dChidG.at(table.at(k)) * n.dGdr.at(k);
            }
#else
            for (size_t k = 0; k < numSymmetryFunctions; ++k)
            {
                dChidr += dChidG.at(k) * n.dGdr.at(k);
            }
#endif
        }
    }
    return dChidr;
}

vector<string> Atom::getForcesLines() const
{
    vector<string> v;
    for (size_t i = 0; i < 3; ++i)
    {
        v.push_back(strpr("%10zu %10zu %16.8E %16.8E\n",
                          indexStructure,
                          index,
                          fRef[i],
                          f[i]));
    }

    return v;
}

string Atom::getChargeLine() const
{
    return strpr("%10zu %10zu %16.8E %16.8E\n",
                 indexStructure,
                 index,
                 chargeRef,
                 charge);
}

vector<string> Atom::info() const
{
    vector<string> v;

    v.push_back(strpr("********************************\n"));
    v.push_back(strpr("ATOM                            \n"));
    v.push_back(strpr("********************************\n"));
    v.push_back(strpr("hasNeighborList                : %d\n", hasNeighborList));
    v.push_back(strpr("hasSymmetryFunctions           : %d\n", hasSymmetryFunctions));
    v.push_back(strpr("hasSymmetryFunctionDerivatives : %d\n", hasSymmetryFunctionDerivatives));
    v.push_back(strpr("useChargeNeuron                : %d\n", useChargeNeuron));
    v.push_back(strpr("index                          : %d\n", index));
    v.push_back(strpr("indexStructure                 : %d\n", indexStructure));
    v.push_back(strpr("tag                            : %" PRId64 "\n", tag));
    v.push_back(strpr("element                        : %d\n", element));
    v.push_back(strpr("numNeighbors                   : %d\n", numNeighbors));
    v.push_back(strpr("numNeighborsUnique             : %d\n", numNeighborsUnique));
    v.push_back(strpr("numSymmetryFunctions           : %d\n", numSymmetryFunctions));
    v.push_back(strpr("energy                         : %16.8E\n", energy));
    v.push_back(strpr("chi                            : %16.8E\n", chi));
    v.push_back(strpr("charge                         : %16.8E\n", charge));
    v.push_back(strpr("chargeRef                      : %16.8E\n", chargeRef));
    v.push_back(strpr("r                              : %16.8E %16.8E %16.8E\n", r[0], r[1], r[2]));
    v.push_back(strpr("f                              : %16.8E %16.8E %16.8E\n", f[0], f[1], f[2]));
    v.push_back(strpr("fRef                           : %16.8E %16.8E %16.8E\n", fRef[0], fRef[1], fRef[2]));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("neighborsUnique            [*] : %d\n", neighborsUnique.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < neighborsUnique.size(); ++i)
    {
        v.push_back(strpr("%29d  : %d\n", i, neighborsUnique.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("numNeighborsPerElement     [*] : %d\n", numNeighborsPerElement.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < numNeighborsPerElement.size(); ++i)
    {
        v.push_back(strpr("%29d  : %d\n", i, numNeighborsPerElement.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("numSymmetryFunctionDeriv.  [*] : %d\n", numSymmetryFunctionDerivatives.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < numSymmetryFunctionDerivatives.size(); ++i)
    {
        v.push_back(strpr("%29d  : %d\n", i, numSymmetryFunctionDerivatives.at(i)));
    }
#ifndef N2P2_NO_SF_CACHE
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("cacheSizePerElement        [*] : %d\n", cacheSizePerElement.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < cacheSizePerElement.size(); ++i)
    {
        v.push_back(strpr("%29d  : %d\n", i, cacheSizePerElement.at(i)));
    }
#endif
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("G                          [*] : %d\n", G.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < G.size(); ++i)
    {
        v.push_back(strpr("%29d  : %16.8E\n", i, G.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("dEdG                       [*] : %d\n", dEdG.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < dEdG.size(); ++i)
    {
        v.push_back(strpr("%29d  : %16.8E\n", i, dEdG.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("dQdG                       [*] : %d\n", dQdG.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < dQdG.size(); ++i)
    {
        v.push_back(strpr("%29d  : %16.8E\n", i, dQdG.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
#ifdef N2P2_FULL_SFD_MEMORY
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("dGdxia                     [*] : %d\n", dGdxia.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < dGdxia.size(); ++i)
    {
        v.push_back(strpr("%29d  : %16.8E\n", i, dGdxia.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
#endif
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("dGdr                       [*] : %d\n", dGdr.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < dGdr.size(); ++i)
    {
        v.push_back(strpr("%29d  : %16.8E %16.8E %16.8E\n", i, dGdr.at(i)[0], dGdr.at(i)[1], dGdr.at(i)[2]));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("neighbors                  [*] : %d\n", neighbors.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        v.push_back(strpr("%29d  :\n", i));
        vector<string> vn = neighbors[i].info();
        v.insert(v.end(), vn.begin(), vn.end());
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("********************************\n"));

    return v;
}

Atom::Neighbor::Neighbor() : index      (0                      ),
                             tag        (0                      ),
                             element    (0                      ),
                             d          (0.0                    )
{
}

bool Atom::Neighbor::operator==(Atom::Neighbor const& rhs) const
{
    if (element != rhs.element) return false;
    if (d       != rhs.d      ) return false;
    return true;
}

bool Atom::Neighbor::operator<(Atom::Neighbor const& rhs) const
{
    // sorting according to elements deactivated
    // TODO: resolve this issue for nnp-sfclust
    // TODO: this breaks pynnp CI tests in test_Neighbor.py
    //if      (element < rhs.element) return true;
    //else if (element > rhs.element) return false;
    if      (d       < rhs.d      ) return true;
    else if (d       > rhs.d      ) return false;
    return false;
}

vector<string> Atom::Neighbor::info() const
{
    vector<string> v;

    v.push_back(strpr("********************************\n"));
    v.push_back(strpr("NEIGHBOR                        \n"));
    v.push_back(strpr("********************************\n"));
    v.push_back(strpr("index                          : %d\n", index));
    v.push_back(strpr("tag                            : %" PRId64 "\n", tag));
    v.push_back(strpr("element                        : %d\n", element));
    v.push_back(strpr("d                              : %16.8E\n", d));
    v.push_back(strpr("dr                             : %16.8E %16.8E %16.8E\n", dr[0], dr[1], dr[2]));
    v.push_back(strpr("--------------------------------\n"));
#ifndef N2P2_NO_SF_CACHE
    v.push_back(strpr("cache                      [*] : %d\n", cache.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < cache.size(); ++i)
    {
        v.push_back(strpr("%29d  : %16.8E\n", i, cache.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
#endif
    v.push_back(strpr("dGdr                       [*] : %d\n", dGdr.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < dGdr.size(); ++i)
    {
        v.push_back(strpr("%29d  : %16.8E %16.8E %16.8E\n", i, dGdr.at(i)[0], dGdr.at(i)[1], dGdr.at(i)[2]));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("********************************\n"));

    return v;
}

#include <iostream>
#include "citationHMM.hpp"

namespace chmmpp {

namespace {

void local_learn_citation(HMM &hmm, const std::vector<std::vector<int> > &obs, const std::vector<int> &numZeros, const double convergence_tolerance)
{
    return;
}

}

void citationHMM::learn_citation(const std::vector<std::vector<int>> &obs)
{
    return;
}

void citationHMM::learn_citation(const std::vector<int> &obs)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    learn_citation(newObs);
}

}  // namespace chmmpp

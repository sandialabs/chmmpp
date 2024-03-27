#pragma once

#include <vector>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

// TODO - Document the semantics of these inference methods.

void aStar(const HMM &hmm, const std::vector<int> &observations, std::vector<int> &hidden_states, double &logProb);

// WEH - This is an application-specific A* implementation?
void aStar_numZeros(const HMM& hmm, const std::vector<int> &observations, std::vector<int> &hidden_states, double &logProb, const int numZeros);

void aStarOracle(const HMM& hmm, const std::vector<int> &observations, std::vector<int>& hidden_states, double &logProb, const std::function<bool(std::vector<int>)> &constraintOracle);

// WEH - This is an application-specific A* implementation?
void aStarMult_numZeros(const HMM& hmm, const std::vector<int> &observations, std::vector<std::vector<int>> &hidden_states, double &logProb, const int numZeros, const int numSolns);

void aStarMultOracle(const HMM& hmm, const std::vector<int> &observations, std::vector<std::vector<int>> &hidden_states, double &logProb, const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns);


} // namespace chmmpp

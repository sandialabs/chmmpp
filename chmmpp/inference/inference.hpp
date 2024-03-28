#pragma once

#include <functional>
#include <vector>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

// TODO - Document the semantics of these inference methods.

// Using a linear program to compute the maximum a posteriori probability (MAP) estimate of the most
// likely sequence of hidden states WEH - Is 'inference' redundant here?  We could use a 'inference'
// namespace for all of these functions in this header...
void lp_map_inference(const HMM &hmm, const std::vector<int> &observations,
                      std::vector<int> &hidden_states, double &logProb);

void aStar(const HMM &hmm, const std::vector<int> &observations, std::vector<int> &hidden_states,
           double &logProb); // Can be removed -- viterbi is faster and gives the same answer
void viterbi(const HMM &hmm, const std::vector<int> &observations, std::vector<int> &hidden_states,
           double &logProb); //This should give the same answer as aStar. 

// WEH - This is an application-specific A* implementation?
void aStar_numZeros(const HMM &hmm, const std::vector<int> &observations,
                    std::vector<int> &hidden_states, double &logProb, const int numZeros);

void aStarOracle(const HMM &hmm, const std::vector<int> &observations,
                 std::vector<int> &hidden_states, double &logProb,
                 const std::function<bool(std::vector<int>&)> &constraintOracle);

// WEH - This is an application-specific A* implementation?
void aStarMult_numZeros(const HMM &hmm, const std::vector<int> &observations,
                        std::vector<std::vector<int>> &hidden_states, double &logProb,
                        const int numZeros, const int numSolns);

void aStarMultOracle(const HMM &hmm, const std::vector<int> &observations,
                     std::vector<std::vector<int>> &hidden_states, double &logProb,
                     const std::function<bool(std::vector<int>&)> &constraintOracle,
                     const int numSolns);

}  // namespace chmmpp

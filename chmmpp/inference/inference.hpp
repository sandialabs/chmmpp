#pragma once

#include <functional>
#include <vector>
#include <chmmpp/HMM.hpp>

namespace chmmpp {

// TODO - Document the semantics of these inference methods.

// Using a linear program to compute the maximum a posteriori probability (MAP) estimate of the most
// likely sequence of hidden states
//
// WEH - Is 'inference' redundant here?  We could use a 'inference'
// namespace for all of these functions in this header...
void lp_map_inference(const HMM &hmm, const std::vector<int> &observations,
                      std::vector<int> &hidden_states, double &logProb, const Options &options);

void aStar(const HMM &hmm, const std::vector<int> &observations, std::vector<int> &hidden_states,
           double &logProb);

void viterbi(const HMM &hmm, const std::vector<int> &observations, std::vector<int> &hidden_states,
             double &logProb);

void aStarOracle(const HMM &hmm, const std::vector<int> &observations,
                 std::vector<int> &hidden_states, double &logProb,
                 const std::shared_ptr<Constraint_Oracle_Base>& constraint_oracle);

void aStarMultOracle(const HMM &hmm, const std::vector<int> &observations,
                     std::vector<std::vector<int>> &hidden_states, std::vector<double> &logProb,
                     const std::shared_ptr<Constraint_Oracle_Base>& constraint_oracle,
                     const int numSolns, const Options &options);
void aStarMultOracle(const HMM &hmm, const std::vector<int> &observations,
                     std::vector<std::vector<int>> &hidden_states, std::vector<double> &logProb,
                     const std::shared_ptr<Constraint_Oracle_Base>& constraint_oracle,
                     const int numSolns, unsigned int max_iterations);

}  // namespace chmmpp

#pragma once

#include <functional>
#include <vector>
#include <chmmpp/HMM.hpp>

namespace chmmpp {

void learn_unconstrained(HMM &hmm, const std::vector<int> &obs);
void learn_unconstrained(HMM &hmm, const std::vector<std::vector<int> > &obs);

void learn_numZeros(HMM &hmm, const std::vector<int> &obs, const int numZeros,
                    const double eps = 10E-6);
void learn_numZeros(HMM &hmm, const std::vector<std::vector<int> > &obs,
                    const std::vector<int> &numZeros, const double eps = 10E-6);

void learn_stochastic(HMM &hmm, const std::vector<std::vector<int> > &obs,
                      const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
                      const Options &options);
void learn_stochastic(HMM &hmm, const std::vector<int> &obs,
                      const std::function<bool(std::vector<int>)> &constraintOracle,
                      const Options &options);

void learn_hardEM(HMM &hmm, const std::vector<std::vector<int> > &obs,
                  const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
                  const int numSolns, const Options &options);
void learn_hardEM(HMM &hmm, const std::vector<int> &obs,
                  const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns,
                  const Options &options);

}  // namespace chmmpp

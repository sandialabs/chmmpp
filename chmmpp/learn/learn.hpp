#pragma once

#include <functional>
#include <vector>
#include <chmmpp/HMM.hpp>

namespace chmmpp {

void estimate_hmm(HMM &hmm, const std::vector<std::vector<int> > &obs,
                  const std::vector<std::vector<int> > &hid);

void learn_unconstrained(HMM &hmm, const std::vector<int> &obs);
void learn_unconstrained(HMM &hmm, const std::vector<std::vector<int> > &obs);

#if 0
void learn_numZeros(HMM &hmm, const std::vector<int> &obs, const int numZeros,
                    const double eps = 10E-6);
void learn_numZeros(HMM &hmm, const std::vector<std::vector<int> > &obs,
                    const std::vector<int> &numZeros, const double eps = 10E-6);
#endif

void learn_batch(HMM &hmm, const std::vector<std::vector<int>> &obs, 
                const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle,
                const std::function<std::vector<std::vector<std::vector<int>>>(
                    HMM&, const int&, const int&, const std::vector<std::vector<int>>&, 
                    const std::vector<std::function<bool(std::vector<int>&)> >&
                )> generator,
                const Options& options);

void learn_stochastic(HMM &hmm, const std::vector<std::vector<int> > &obs,
                      const std::vector<std::function<bool(std::vector<int> &)> > &constraintOracle,
                      const Options &options);
void learn_stochastic(HMM &hmm, const std::vector<int> &obs,
                      const std::function<bool(std::vector<int> &)> &constraintOracle,
                      const Options &options);

void learn_hardEM(HMM &hmm, const std::vector<std::vector<int> > &obs,
                  const std::vector<std::function<bool(std::vector<int> &)> > &constraintOracle,
                  const int numSolns, const Options &options);
void learn_hardEM(HMM &hmm, const std::vector<int> &obs,
                  const std::function<bool(std::vector<int> &)> &constraintOracle,
                  const int numSolns, const Options &options);

void learn_semisupervised_hardEM(HMM &hmm, const std::vector<std::vector<int> > &supervisedObs,
                                 const std::vector<std::vector<int> > &supervisedHidden,
                                 const std::vector<std::vector<int> > &unsupervisedObs,
                                 const std::function<bool(std::vector<int> &)> &constraintOracle,
                                 bool partialOracle, const Options &options);

}  // namespace chmmpp

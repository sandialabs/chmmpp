#pragma once

#include <functional>
#include <vector>
#include <chmmpp/HMM.hpp>

namespace chmmpp {

void estimate_hmm(HMM &hmm, const std::vector<std::vector<int> > &obs,
                  const std::vector<std::vector<int> > &hid);

void learn_unconstrained(HMM &hmm, const std::vector<int> &obs);
void learn_unconstrained(HMM &hmm, const std::vector<std::vector<int> > &obs);

void learn_batch(HMM &hmm, 
                const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle,
                const std::vector<std::vector<int>> &obs, 
                const std::function<std::vector<std::vector<std::vector<int>>> (
                        HMM&, const std::vector<std::function<bool(std::vector<int> &)>>&,
                        const std::vector<std::vector<int>>&, 
                        const int&, const int&
                    )> generator,
                const Options& options); 

void learn_semisupervised_hardEM(HMM &hmm, const std::vector<std::vector<int> > &supervisedObs,
                                 const std::vector<std::vector<int> > &supervisedHidden,
                                 const std::vector<std::vector<int> > &unsupervisedObs,
                                 const std::function<bool(std::vector<int> &)> &constraintOracle,
                                 bool partialOracle, const Options &options);
}  // namespace chmmpp

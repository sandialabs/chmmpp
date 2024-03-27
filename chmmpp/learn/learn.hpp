#pragma once

#include <vector>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

    void learn_unconstrained(HMM &hmm, const std::vector<int> &obs, const double eps = 10E-6);
    void learn_unconstrained(HMM &hmm, const std::vector<std::vector<int> > &obs, const double eps = 10E-6); 

    void learn_numZeros(HMM &hmm, const std::vector<int> &obs, const int numZeros, const double eps = 10E-6);
    void learn_numZeros(HMM &hmm, const std::vector<std::vector<int> > &obs, const std::vector<int> &numZeros,
               const double eps = 10E-6);

    void learn_stochastic(HMM &hmm, const std::vector<std::vector<int> > &obs,
               const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
               const double eps = 10E-6, const int C = 10E4);
    void learn_stochastic(HMM &hmm, const std::vector<int> &obs,
               const std::function<bool(std::vector<int>)> &constraintOracle,
               const double eps = 10E-6, const int C = 10E4);

    void learn_hardEM(HMM &hmm, const std::vector<std::vector<int> > &obs,
                   const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
                   const int numSolns = 1, const double eps = 10E-6);
    void learn_hardEM(HMM &hmm, const std::vector<int> &obs,
                   const std::function<bool(std::vector<int>)> &constraintOracle,
                   const int numSolns = 1, const double eps = 10E-6);

}  // namespace chmmpp

#pragma once

#include <functional>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

class HMM_inference : public HMM {
   public:
    HMM_inference(long int seed = time(NULL)) : HMM(seed) {}

    HMM_inference(const std::vector<std::vector<double> > &inputA,
                  const std::vector<double> &inputS,
                  const std::vector<std::vector<double> > &inputE, long int seed = time(NULL))
        : HMM(inputA, inputS, inputE, seed)
    {
    }

    // Inference
    std::vector<int> aStar(const std::vector<int> &observations, double &logProb) const;
    std::vector<int> aStar(const std::vector<int> &observations, double &logProb,
                           const int numZeros) const;
    std::vector<int> aStarOracle(
        const std::vector<int> &observations, double &logProb,
        const std::function<bool(std::vector<int>)> &constraintOracle) const;
    std::vector<std::vector<int> > aStarMult(const std::vector<int> &observations, double &logProb,
                                             const int numZeros, const int numSolns) const;
    std::vector<std::vector<int> > aStarMult(
        const std::vector<int> &observations, double &logProb,
        const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns) const;

    // Learning Algorithms
    void learn(const std::vector<int> &obs, const int numZeros, const double eps = 10E-6);
    void learn(const std::vector<std::vector<int> > &obs, const std::vector<int> &numZeros,
               const double eps = 10E-6);
    void learn(const std::vector<int> &obs, const double eps = 10E-6);
    void learn(const std::vector<std::vector<int> > &obs, const double eps = 10E-6);
    void learn(const std::vector<std::vector<int> > &obs,
               const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
               const double eps = 10E-6, const int C = 10E4);
    void learnHard(const std::vector<std::vector<int> > &obs,
                   const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
                   const double eps = 10E-6, int numSolns = 1);
};

}  // namespace chmmpp

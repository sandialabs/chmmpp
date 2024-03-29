#pragma once

#include <functional>
#include <chmmpp/HMM.hpp>

namespace chmmpp {

//
// A base class that supports various methods for constrained inference.
//
class CHMM : public Options {
   public:
    HMM hmm;
    std::function<bool(std::vector<int> &)> constraintOracle;

   public:
    void initialize(const HMM &_hmm) { hmm = _hmm; }

    void initialize(const std::vector<std::vector<double>> &inputA,
                    const std::vector<double> &inputS,
                    const std::vector<std::vector<double>> &inputE, long int seed)
    {
        hmm.initialize(inputA, inputS, inputE, seed);
    }

    virtual void initialize_from_file(const std::string &json_filename)
    {
        hmm.initialize_from_file(json_filename);
    }

    virtual void initialize_from_string(const std::string &json_string)
    {
        hmm.initialize_from_string(json_string);
    }

    virtual double logProb(const std::vector<int> obs, const std::vector<int> guess) const
    {
        return hmm.logProb(obs, guess);
    }

    //
    // inference methods
    //

    // aStar using a constraintOracle function to identify feasible solutions
    void aStar(const std::vector<int> &observations, std::vector<int> &hidden_states,
               double &logProb);

    // aStar using a constraintOracle function to identify feasible solutions, generating
    // multiple solutions
    //
    //  Options
    //      max_iterations (int):   Stop learning if number of iterations equals this threshold.
    //                              No threshold if this is 0 (Default: 0).
    //
    void aStarMult(const std::vector<int> &observations,
                   std::vector<std::vector<int>> &hidden_states, std::vector<double> &logProb,
                   const int numSolns);

    // Optimize using an mixed-integer programming formulation that expresses application
    // constraints
    virtual void mip_map_inference(const std::vector<int> &observations,
                                   std::vector<int> &hidden_states, double &logProb);

    //
    // learning methods
    //

    void learn_stochastic(
        HMM &hmm, const std::vector<std::vector<int>> &obs,
        const std::vector<std::function<bool(std::vector<int>)>> &constraintOracle,
        const double eps = 10E-6, const int C = 10E4);
    void learn_stochastic(HMM &hmm, const std::vector<int> &obs,
                          const std::function<bool(std::vector<int>)> &constraintOracle,
                          const double eps = 10E-6, const int C = 10E4);

    void learn_hardEM(HMM &hmm, const std::vector<std::vector<int>> &obs,
                      const std::vector<std::function<bool(std::vector<int>)>> &constraintOracle,
                      const int numSolns = 1, const double eps = 10E-6);
    void learn_hardEM(HMM &hmm, const std::vector<int> &obs,
                      const std::function<bool(std::vector<int>)> &constraintOracle,
                      const int numSolns = 1, const double eps = 10E-6);
};

}  // namespace chmmpp

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
    bool partialOracle = false;  // True if constraintOracle can be applied to partial sequences

   public:
    void initialize(const HMM &_hmm) { hmm = _hmm; }

    void initialize(const std::vector<std::vector<double>> &inputA,
                    const std::vector<double> &inputS,
                    const std::vector<std::vector<double>> &inputE)
    {
        hmm.initialize(inputA, inputS, inputE);
    }

    void set_seed(long int seed) { hmm.set_seed(seed); }
    void reset_rng() { hmm.reset_rng(); }

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

    virtual void print() const { hmm.print(); }

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

    // TODO
    //
    //  Options
    //      convergence_tolerance (double):     Stop learning when difference in model parameters
    //                                          falls below this threshold (Default: 10E-6).
    //      C (unsigned int):                   TODO
    //
    void learn_stochastic(const std::vector<std::vector<int>> &obs);

    // TODO
    //
    //  Options
    //      convergence_tolerance (double):     Stop learning when difference in model parameters
    //                                          falls below this threshold (Default: 10E-6).
    //      C (unsigned int):                   TODO
    //
    void learn_stochastic(const std::vector<int> &obs);

    // TODO
    //
    //  Options
    //      convergence_tolerance (double):     Stop learning when difference in model parameters
    //                                          falls below this threshold (Default: 10E-6).
    //
    void learn_hardEM(const std::vector<std::vector<int>> &obs, const int numSolns = 1);

    // TODO
    //
    //  Options
    //      convergence_tolerance (double):     Stop learning when difference in model parameters
    //                                          falls below this threshold (Default: 10E-6).
    //
    void learn_hardEM(const std::vector<int> &obs, const int numSolns = 1);

    // CLM - IN PROGRESS
    //
    //   Options
    //       convergence_tolerance (double):     Stop learning when difference in model parameters
    //                                           falls below this threshold (Default: 10E-6).
    //       gamma (double):                     Percent unsupervised solutions are under-weighted
    //
    //
    void learn_semisupervised_hardEM(const std::vector<std::vector<int>> &supervisedObs,
                                     const std::vector<std::vector<int>> &supervisedHidden,
                                     const std::vector<std::vector<int>> &unsupervisedObs);
};

}  // namespace chmmpp

#pragma once

#include <functional>
#include <chmmpp/HMM.hpp>
#include <chmmpp/inference/inference.hpp>
#include <chmmpp/learn/learn.hpp>

namespace chmmpp {

//
// A base class that supports various methods for constrained inference.
//
class CHMM : public Options {
   public:
    HMM hmm;
    
    //Optional Variables
    std::shared_ptr<Generator_Base> generator_stochastic;
    std::shared_ptr<Generator_Base> generator_hardEM;
    std::shared_ptr<Generator_Base> generator_MIP; 
    std::shared_ptr<Constraint_Oracle_Base> constraint_oracle;

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

    virtual void run(const int &T, std::vector<int> &observedStates, std::vector<int> &hiddenStates);

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

    //Most general learning method called by all the others
    void learn_batch(const std::vector<std::vector<int>> &obs, Generator_Base &generator);
    void learn_batch(const std::vector<int> &obs, Generator_Base &generator);

    void learn_stochastic(const std::vector<std::vector<int>> &obs);
    void learn_stochastic(const std::vector<int> &obs);

    void learn_hardEM(const std::vector<std::vector<int>> &obs);
    void learn_hardEM(const std::vector<int> &obs);

    void learn_MIP(const std::vector<std::vector<int>> &obs);
    void learn_MIP(const std::vector<int> &obs);


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

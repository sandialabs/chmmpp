#include <vector>
#include <random>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

class LearnStochastic {
   public:
    HMM hmm;
    std::mt19937 generator;

    void initialize(HMM &hmm_) { hmm = hmm_; }
    void set_seed(long int);
    long int generate_seed();

    // select == 0
    void learn(const std::vector<std::vector<int> > &obs, double convergence_tolerance,
               unsigned int C, unsigned int max_iterations);
    // select == 1
    void learn1(const std::vector<std::vector<int> > &obs, double convergence_tolerance,
               unsigned int C, unsigned int max_iterations, unsigned int quiet);
    void learn(const std::vector<std::vector<int> > &obs, Options &options);
    void learn(const std::vector<int> &obs, Options &options);

    virtual std::pair<std::vector<int>,double> generate_feasible_hidden(size_t T, const std::vector<int> &obs) = 0;
    virtual std::vector<int> generate_random_feasible_hidden(size_t T, const std::vector<int> &obs, long int seed) = 0; 

    //New from CLM
    //Helper function, not called directly
    void learn_batch2(const std::vector<std::vector<int>> &obs, 
                const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle,
                const std::function<std::vector<std::vector<std::vector<int>>>(
                    HMM&, const int&, const int&, const std::vector<std::vector<int>>&, 
                    const std::vector<std::function<bool(std::vector<int>&)> >&
                )> generator,
                const Options& options);
    
    //These call learn_batch
    void learn_stochastic2(const std::vector<std::vector<int> > &obs,
                      const std::vector<std::function<bool(std::vector<int> &)> > &constraintOracle,
                      const Options &options);
    void learn_stochastic2(const std::vector<int> &obs,
                      const std::function<bool(std::vector<int> &)> &constraintOracle,
                      const Options &options);

    void learn_hardEM2(const std::vector<std::vector<int> > &obs,
                  const std::vector<std::function<bool(std::vector<int> &)> > &constraintOracle,
                  const int numSolns, const Options &options);
    void learn_hardEM2(const std::vector<int> &obs,
                  const std::function<bool(std::vector<int> &)> &constraintOracle,
                  const int numSolns, const Options &options);

    //TODO make a learn_projection method
};

}  // namespace chmmpp

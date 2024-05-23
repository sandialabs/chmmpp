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
};

}  // namespace chmmpp

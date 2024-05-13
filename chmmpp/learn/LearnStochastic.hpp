#include <vector>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

class LearnStochastic {
   public:
    HMM hmm;

    void initialize(HMM &hmm_) { hmm = hmm_; }

    // select == 0
    void learn(const std::vector<std::vector<int> > &obs, double convergence_tolerance,
               unsigned int C, unsigned int max_iterations);
    // select == 1
    void learn1(const std::vector<std::vector<int> > &obs, double convergence_tolerance,
               unsigned int C, unsigned int max_iterations);
    void learn(const std::vector<std::vector<int> > &obs, const Options &options);
    void learn(const std::vector<int> &obs, const Options &options);

    virtual std::pair<std::vector<int>,double> generate_feasible_hidden(size_t T, const std::vector<int> &obs) = 0;
    virtual std::vector<int> generate_random_feasible_hidden(size_t T, const std::vector<int> &obs) = 0;
};

}  // namespace chmmpp

#include <vector>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

class LearnStochastic {
   public:
    HMM hmm;

    void initialize(HMM &hmm_) { hmm = hmm_; }

    void learn(const std::vector<std::vector<int> > &obs, const double convergence_tolerance,
               const int C);
    void learn(const std::vector<std::vector<int> > &obs, const Options &options);
    void learn(const std::vector<int> &obs, const Options &options);

    virtual std::vector<int> generate_feasible_hidden(size_t T, const std::vector<int> &obs) = 0;
};

}  // namespace chmmpp

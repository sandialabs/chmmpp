#include "citationHMM.hpp"
#ifdef WITH_COEK
#    include <chmmpp/inference/LPModel.hpp>
#endif

namespace chmmpp {

#ifdef WITH_COEK
class MIPModel : public LPModel {
   public:
    // virtual void set_options(const Options& options);

    void initialize(const citationHMM& hmm, const std::vector<int>& observations);

    // void optimize(double& log_likelihood, std::vector<int>& hidden_states);

    void collect_solution(std::vector<int>& hidden_states);
};

void MIPModel::initialize(const citationHMM& hmm, const std::vector<int>& observations)
{
#if 0
    // Require the use of binary flow variables (y)
    y_binary = true;
    LPModel::initialize(hmm.hmm, observations);

    // # of times the solution flows through state 0  == numZeros
    auto sum = coek::expression();
    for (auto t : coek::range(Tmax)) {
        if (t == 0)
            sum += y[{t - 1, -1, 0}];
        else {
            for (auto a : coek::range(N))
                if (not(F.find({a, 0}) == F.end())) sum += y[{t - 1, a, 0}];
        }
    }
    model.add(sum == hmm.numZeros);
#endif
}

void MIPModel::collect_solution(std::vector<int>& hidden_states)
{
    LPModel::collect_solution(hidden_states);

    // TODO - more here ?
}
#endif

void citationHMM::mip_map_inference_citation(const std::vector<int>& observations,
                                    std::vector<int>& hidden_states, double& logProb)
{
#ifdef WITH_COEK
    MIPModel model;

    model.set_options(get_options());
    model.initialize(*this, observations);
    model.optimize(logProb, hidden_states);
#else
    hidden_states.resize(hmm.getH());
    logProb = 0;
#endif
}

}  // namespace chmmpp

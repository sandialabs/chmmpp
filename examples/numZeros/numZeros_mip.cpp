#include "numZerosHMM.hpp"
#include <chmmpp/learn/LearnStochastic.hpp>
#ifdef WITH_COEK
#    include <chmmpp/inference/LPModel.hpp>
#endif

namespace chmmpp {

// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------

#ifdef WITH_COEK
class InferenceModel : public LPModel {
   public:
    // virtual void set_options(const Options& options);

    void initialize(const numZerosHMM& nzhmm, const std::vector<int>& observations);

    // void optimize(double& log_likelihood, std::vector<int>& hidden_states);

    void collect_solution(std::vector<int>& hidden_states);
};

void InferenceModel::initialize(const numZerosHMM& nzhmm, const std::vector<int>& observations)
{
    // Require the use of binary flow variables (y)
    y_binary = true;
    LPModel::initialize(nzhmm.hmm, observations);

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
    model.add(sum == nzhmm.numZeros);
}

void InferenceModel::collect_solution(std::vector<int>& hidden_states)
{
    LPModel::collect_solution(hidden_states);

    // TODO - more here ?
}
#endif

void numZerosHMM::mip_map_inference(const std::vector<int>& observations,
                                    std::vector<int>& hidden_states, double& log_likelihood)
{
#ifdef WITH_COEK
    InferenceModel model;

    model.set_options(get_options());
    model.initialize(*this, observations);
    model.optimize(log_likelihood, hidden_states);
#else
    hidden_states.resize(hmm.getH());
    log_likelihood = 0;
#endif
}

// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// Learning
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------

class LearningModel : public InferenceModel {
   public:
    void initialize(numZerosHMM& nzhmm, const std::vector<int>& observations);
};

void LearningModel::initialize(numZerosHMM& nzhmm, const std::vector<int>& observations)
{
    InferenceModel::initialize(nzhmm, observations);

    auto unconstrained_hidden = nzhmm.hmm.generateHidden(observations.size(), observations);

    // Deactivate the old objective
    obj.deactivate();

    // objective
    {
        auto O = coek::expression();
        for (auto t : coek::range(Tmax + 1)) {
            auto lhs = coek::expression();
            for (auto b : coek::range(N)) {
                if (t == 0)
                    lhs = y[{t - 1, -1, b}];
                else {
                    for (auto a : coek::range(N))
                        if (not(F.find({a, b}) == F.end())) lhs += y[{t - 1, a, b}];
                }
            }
            O += (unconstrained_hidden[t] - lhs) * (unconstrained_hidden[t] - lhs);
        }
        model.add_objective(O).sense(model.minimize);
    }
}

class LearnStochastic_numZeros : public LearnStochastic {
   public:
    numZerosHMM& nzhmm;

    LearnStochastic_numZeros(numZerosHMM& nzhmm_) : nzhmm(nzhmm_) { initialize(nzhmm.hmm); }

    std::pair<std::vector<int>,double> generate_feasible_hidden(size_t T, const std::vector<int>& obs);
    std::vector<int> generate_random_feasible_hidden(size_t T, const std::vector<int>& obs);
};

std::pair<std::vector<int>,double> LearnStochastic_numZeros::generate_feasible_hidden(size_t T,
                                                                    const std::vector<int>& obs)
{
    std::vector<int> hidden(obs.size());
    double log_likelihood = 0;
#ifdef WITH_COEK
    InferenceModel model;
    //model.set_options(nzhmm.get_options());
    model.initialize(nzhmm, obs);
    model.optimize(log_likelihood, hidden);
#endif
    return {hidden, log_likelihood};
}

std::vector<int> LearnStochastic_numZeros::generate_random_feasible_hidden(size_t T,
                                                                    const std::vector<int>& obs)
{
    std::vector<int> hidden(obs.size());
    double log_likelihood = 0;
#ifdef WITH_COEK
    LearningModel model;
    //model.set_options(nzhmm.get_options());
    model.initialize(nzhmm, obs);
    model.optimize(log_likelihood, hidden);
#endif
    return hidden;
}

void numZerosHMM::learn_mip(const std::vector<std::vector<int>>& observations)
{
    LearnStochastic_numZeros solver(*this);
    solver.learn(observations, get_options());
    // Copy results back in the nzhmm object
    hmm = solver.hmm;
}

}  // namespace chmmpp

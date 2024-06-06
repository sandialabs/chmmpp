#include "numZerosHMM.hpp"
#include <chmmpp/learn/LearnStochastic.hpp>
#ifdef WITH_COEK
#    include <coek/util/io_utils.hpp>
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

    virtual void print_solution();
};

void InferenceModel::print_solution()
{
#ifdef WITH_COEK
    std::cout << "y size=" << y.size() << std::endl;
    for (auto& it: y) {
        auto [t,a,b] = it.first;
        std::cout << "y id=" << it.second.id() << " t=" << t << " a=" << a << " b=" << b << " value=" << it.second.value() << std::endl;
        }
#endif
}

void InferenceModel::initialize(const numZerosHMM& nzhmm, const std::vector<int>& observations)
{
#ifdef WITH_COEK
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
#endif
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

#ifdef WITH_COEK
class LearningModel : public InferenceModel {
   public:
    std::vector<double> unconstrained_hidden;
    std::map<std::pair<size_t,size_t>,coek::Variable> z;

    void initialize(numZerosHMM& nzhmm, const std::vector<int>& observations);
    void print_solution();
};

void LearningModel::print_solution()
{
    InferenceModel::print_solution();

    std::cout << "z size=" << z.size() << std::endl;
    for (auto& it: z) {
        auto [t,b] = it.first;
        std::cout << "z id=" << it.second.id() << " t=" << t << " b=" << b << " value=" << it.second.value() << std::endl;
        }
}

void LearningModel::initialize(numZerosHMM& nzhmm, const std::vector<int>& observations)
{
    InferenceModel::initialize(nzhmm, observations);

#if 0
    unconstrained_hidden = nzhmm.hmm.generateHidden(observations);
#else
    unconstrained_hidden.resize(observations.size());
    for (auto& val : unconstrained_hidden) {
        val = nzhmm.hmm.getRandom();
    }
#endif

    // Deactivate the old objective
    obj.deactivate();

    // objective
    {
        for (auto t : coek::range(Tmax))
            for (auto b : coek::range(N))
                z[{t,b}] = model.add( coek::variable().bounds(0,1) );

        for (auto t : coek::range(Tmax)) {
            for (auto b : coek::range(N)) {
                auto rhs = coek::expression();
                if (t == 0)
                    rhs = y[{t - 1, -1, b}];
                else {
                    for (auto a : coek::range(N))
                        if (not(F.find({a, b}) == F.end())) rhs += y[{t - 1, a, b}];
                }
                model.add( z[{t,b}] == rhs );
            }
        }

        auto O = coek::expression();
        for (auto t : coek::range(Tmax))
            O += (unconstrained_hidden[t] - z[{t,1}]) * (unconstrained_hidden[t] - z[{t,1}]);
        model.add_objective(O).sense(model.minimize);
    }
}
#endif

class LearnStochastic_numZeros : public LearnStochastic {
   public:
    numZerosHMM& nzhmm;

    LearnStochastic_numZeros(numZerosHMM& nzhmm_) : nzhmm(nzhmm_) { initialize(nzhmm.hmm); }

    std::pair<std::vector<int>,double> generate_feasible_hidden(size_t T, const std::vector<int>& obs);
    std::vector<int> generate_random_feasible_hidden(size_t T, const std::vector<int>& obs, long int seed);
};

std::pair<std::vector<int>,double> LearnStochastic_numZeros::generate_feasible_hidden(size_t T,
                                                                    const std::vector<int>& obs)
{
    std::vector<int> hidden(obs.size());
    double log_likelihood = 0;
#ifdef WITH_COEK
    nzhmm.hmm = hmm;
    InferenceModel model;
    model.set_options(nzhmm.get_options());
    model.initialize(nzhmm, obs);
    //model.print();
    model.optimize(log_likelihood, hidden);
    //model.print_solution();
#endif
    return {hidden, -log_likelihood};
}

std::vector<int> LearnStochastic_numZeros::generate_random_feasible_hidden(size_t T,
                                                                    const std::vector<int>& obs, long int seed)
{
    std::vector<int> hidden(obs.size());
    double log_likelihood = 0;
#ifdef WITH_COEK
    nzhmm.hmm = hmm;
    nzhmm.hmm.set_seed(seed);
    LearningModel model;
    model.set_options(nzhmm.get_options());
    model.initialize(nzhmm, obs);
    //model.print();
    model.optimize(log_likelihood, hidden);
    //model.print_solution();

#if 0
    std::cout << "Random Hidden:           " << model.unconstrained_hidden << std::endl;
    std::cout << "Closest Feasible Hidden: " << hidden << std::endl;
    std::cout << std::endl;
#endif
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

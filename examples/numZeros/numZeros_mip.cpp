#include "numZerosHMM.hpp"
#include "numZeros_mip.hpp"
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

//#ifdef WITH_COEK

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
//#endif

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


//#ifdef WITH_COEK

void LearningModel::print_solution()
{
    InferenceModel::print_solution();

    std::cout << "z size=" << z.size() << std::endl;
    for (auto& it: z) {
        auto [t,b] = it.first;
        std::cout << "z id=" << it.second.id() << " t=" << t << " b=" << b << " value=" << it.second.value() << std::endl;
        }
}

void LearningModel::initialize(numZerosHMM& nzhmm, const std::vector<int>& observations, const std::vector<int>& unconstrained_hidden)
{
    nzhmm.hmm.print();
    std::cout << std::endl;
    InferenceModel::initialize(nzhmm, observations);

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

//#endif



}  // namespace chmmpp





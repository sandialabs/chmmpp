#include "syntheticCitationHMM.hpp"
#include "syntheticCitation_mip.hpp"
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

void InferenceModel::initialize(const syntheticCitationHMM& schmm, const std::vector<int>& observations)
{
#ifdef WITH_COEK
    // Require the use of binary flow variables (y)
    y_binary = true;
    LPModel::initialize(schmm.hmm, observations);

    // # of times the solution flows through state 0  == numZeros
    /*auto sum = coek::expression();
    for (auto t : coek::range(Tmax)) {
        if (t == 0)
            sum += y[{t - 1, -1, 0}];
        else {
            for (auto a : coek::range(N))
                if (not(F.find({a, 0}) == F.end())) sum += y[{t - 1, a, 0}];
        }
    }
    model.add(sum == nzhmm.numZeros);*/

    for(auto h : coek::range(N)) {
        auto sum = coek::expression();

        //start
        sum += y[{-1,-1,h}];

        for(auto g : coek::range(N)) {
            if(h != g) {
                for(auto t : coek::range(Tmax-1)) {
                    if (not(F.find({g,h}) == F.end())) sum += y[{t,g,h}];
                }
            }
        }

        model.add(sum <= 1);
    }
#endif
}

void InferenceModel::collect_solution(std::vector<int>& hidden_states)
{
    #ifdef WITH_COEK
    LPModel::collect_solution(hidden_states);

    // TODO - more here ?
    #endif
}

/*void syntheticCitationHMM::mip_map_inference(const std::vector<int>& observations,
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
}*/

// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// Learning
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------


void LearningModel::print_solution()
{
    #ifdef WITH_COEK
    InferenceModel::print_solution();

    std::cout << "z size=" << z.size() << std::endl;
    for (auto& it: z) {
        auto [t,b] = it.first;
        std::cout << "z id=" << it.second.id() << " t=" << t << " b=" << b << " value=" << it.second.value() << std::endl;
    }
    #endif
}

void LearningModel::initialize(syntheticCitationHMM& schmm, const std::vector<int>& observations, const std::vector<int>& unconstrained_hidden)
{
    #ifdef WITH_COEK
    InferenceModel::initialize(schmm, observations);

    // Deactivate the old objective
    obj.deactivate();

    // objective
    {
        for (auto t : coek::range(Tmax))
            for (auto h : coek::range(N))
                z[{t,h}] = model.add( coek::variable().bounds(0,1) );

        for (auto t : coek::range(Tmax)) {
            for (auto h : coek::range(N)) {
                auto rhs = coek::expression();
                if (t == 0)
                    rhs = y[{t - 1, -1, h}];
                else {
                    for (auto a : coek::range(N))
                        if (not(F.find({a, h}) == F.end())) rhs += y[{t - 1, a, h}];
                }
                model.add( z[{t,h}] == rhs );
            }
        }

        auto O = coek::expression();
        for (auto t : coek::range(Tmax))
            //O += (unconstrained_hidden[t] - z[{t,h}]) * (unconstrained_hidden[t] - z[{t,h}]);
            O += z[{t,unconstrained_hidden[t]}];
        model.add_objective(O).sense(model.maximize);
    }
    #endif
}



std::vector<std::vector<std::vector<int>>> Generator_MIP_SyntheticCitation::operator()(
    HMM &hmm, const std::vector<std::vector<int>>& obs
) const
{
    std::vector<std::vector<std::vector<int>>> output(obs.size());
    for(size_t r = 0; r < obs.size(); ++r) {
        for(size_t b = 0; b < num_solutions; ++b) {
            
            auto hidden = hmm.generateHidden(obs[r]);
            #ifdef WITH_COEK
                LearningModel model;
                syntheticCitationHMM schmm;
                schmm.initialize(hmm);
                model.set_options(schmm.get_options());
                model.initialize(schmm, obs[r], hidden);
                double log_likelihood;
                model.optimize(log_likelihood, hidden);
            #endif

            output[r].push_back(hidden);
        }
    }

    return output;
}

}  // namespace chmmpp





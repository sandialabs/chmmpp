#include "numZerosHMM.hpp"
#ifdef WITH_COEK
#include <chmmpp/inference/LPModel.hpp>
#endif

namespace chmmpp {

class MIPModel : public LPModel 
{
public:

    //virtual void set_options(const Options& options);

    void initialize(const numZerosHMM& hmm, const std::vector<int>& observations);

    //void optimize(double& log_likelihood, std::vector<int>& hidden_states);

    void collect_solution(std::vector<int>& hidden_states);

};

void MIPModel::initialize(const numZerosHMM& hmm, const std::vector<int>& observations)
{
LPModel::initialize(hmm.hmm, observations);

// TODO - more here
}

void MIPModel::collect_solution(std::vector<int>& hidden_states)
{
LPModel::collect_solution(hidden_states);

// TODO - more here
}


void numZerosHMM::mip_map_inference(const std::vector<int> &observations, std::vector<int> &hidden_states, double &logProb)
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

} // namespace chmmpp

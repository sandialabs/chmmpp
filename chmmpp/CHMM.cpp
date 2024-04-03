#include "CHMM.hpp"
#include "inference/inference.hpp"
#include "learn/learn.hpp"

namespace chmmpp {

void CHMM::aStar(const std::vector<int> &observations, std::vector<int> &hidden_states,
                 double &logProb)
{
    chmmpp::aStarOracle(hmm, observations, hidden_states, logProb, constraintOracle);
}

void CHMM::aStarMult(const std::vector<int> &observations,
                     std::vector<std::vector<int>> &hidden_states, std::vector<double> &logProb,
                     const int numSolns)
{
    chmmpp::aStarMultOracle(hmm, observations, hidden_states, logProb, constraintOracle, numSolns,
                            this->get_options());
}

void CHMM::mip_map_inference(const std::vector<int> &observations, std::vector<int> &hidden_states,
                             double &logProb)
{
    hidden_states.resize(hmm.getH());
    logProb = 0;
}

void CHMM::learn_stochastic(const std::vector<std::vector<int>> &obs)
{
    std::vector<std::function<bool(std::vector<int>&)>> oracles(obs.size());
    for (size_t i=0; i<obs.size(); i++)
        oracles[i] = constraintOracle;
    chmmpp::learn_stochastic(hmm, obs, oracles, this->get_options());
};

void CHMM::learn_stochastic(const std::vector<int> &obs)
{
    chmmpp::learn_stochastic(hmm, obs, constraintOracle, this->get_options());
}

void CHMM::learn_hardEM(const std::vector<std::vector<int>> &obs, int numSolns)
{
    std::vector<std::function<bool(std::vector<int>&)>> oracles;
    for (size_t i=0; i<obs.size(); i++)
        oracles.push_back(constraintOracle);
    chmmpp::learn_hardEM(hmm, obs, oracles, numSolns, this->get_options());
}

void CHMM::learn_hardEM(const std::vector<int> &obs, int numSolns)
{
    chmmpp::learn_hardEM(hmm, obs, constraintOracle, numSolns, this->get_options());
}

void CHMM::learn_semisupervised_hardEM(const std::vector< std::vector<int> > &supervisedObs, 
                                 const std::vector< std::vector<int> > &supervisedHidden, 
                                 const std::vector< std::vector<int> > &unsupervisedObs) 
{
    chmmpp::learn_semisupervised_hardEM(hmm, supervisedObs, supervisedHidden, unsupervisedObs, this->get_options());
}

}  // namespace chmmpp

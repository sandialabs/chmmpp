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

void CHMM::learn_batch(const std::vector<std::vector<int>> &obs, 
                const std::function<std::vector<std::vector<std::vector<int>>> (
                        HMM&, const std::vector<std::function<bool(std::vector<int> &)>>&,
                        const std::vector<std::vector<int>>&, 
                        const int&, const int&
                    )> generator)
{
    std::vector<std::function<bool(std::vector<int> &)>> oracles(obs.size());
    for (size_t i = 0; i < obs.size(); i++) oracles[i] = constraintOracle;
    chmmpp::learn_batch(hmm, oracles, obs, generator, this->get_options());
} 

void CHMM::learn_stochastic(const std::vector<std::vector<int>> &obs)
{
    learn_batch(obs, generator_stochastic);
};

void CHMM::learn_stochastic(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_stochastic(obsVec);
}

void CHMM::learn_hardEM(const std::vector<std::vector<int>> &obs)
{
    learn_batch(obs, generator_hardEM);
}

void CHMM::learn_hardEM(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_hardEM(obsVec);
}

//Give an IP generator for this to work
void CHMM::learn_IP(const std::vector<std::vector<int>> &obs)
{
    learn_batch(obs, generator_IP);
}

void CHMM::learn_hardEM(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_IP(obsVec);
}

void CHMM::learn_semisupervised_hardEM(const std::vector<std::vector<int>> &supervisedObs,
                                       const std::vector<std::vector<int>> &supervisedHidden,
                                       const std::vector<std::vector<int>> &unsupervisedObs)
{
    chmmpp::learn_semisupervised_hardEM(hmm, supervisedObs, supervisedHidden, unsupervisedObs,
                                        this->constraintOracle, this->partialOracle,
                                        this->get_options());
}

}  // namespace chmmpp

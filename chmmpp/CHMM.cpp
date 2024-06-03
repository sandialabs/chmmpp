#include "CHMM.hpp"
#include "inference/inference.hpp"

namespace chmmpp {

//TODO - add max_iterations
void CHMM::run(const int &T, std::vector<int> &observedStates, std::vector<int> &hiddenStates)
{
    while(true) {
        hmm.run(T,observedStates,hiddenStates);
        if(constraintOracle(hiddenStates)) {
            break;
        }
    }
}

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

void CHMM::learn_batch(const std::vector<std::vector<int>> &obs, Generator_Base &generator)
{
    chmmpp::learn_batch(hmm, obs, generator, this->get_options());
}
void CHMM::learn_batch(const std::vector<int> &obs, Generator_Base &generator)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_batch(obsVec, generator);
}

void CHMM::learn_stochastic(const std::vector<std::vector<int>> &obs) {
    if(generator_stochastic) {
        learn_batch(obs, *generator_stochastic);
    }
    else if(constraint_oracle) {
        Generator_Stochastic _generator_stochastic(constraint_oracle);
        learn_batch(obs,_generator_stochastic);
    }
    else {
        std::cout << "ERROR: In learn_stochastic either generator_stochastic or constraint_oracle must be defined." << std::endl;
    }
}
void CHMM::learn_stochastic(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_stochastic(obsVec);
}

void CHMM::learn_hardEM(const std::vector<std::vector<int>> &obs) {
    if(generator_hardEM) {
        learn_batch(obs, *generator_hardEM);
    }
    else if(constraint_oracle) {
        Generator_HardEM _generator_hardEM(constraint_oracle);
        learn_batch(obs,_generator_hardEM);
    }
    else {
        std::cout << "ERROR: In learn_hardEM either generator_hardEM or constraint_oracle must be defined." << std::endl;
    }
}
void CHMM::learn_hardEM(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_hardEM(obsVec);
}

void CHMM::learn_MIP(const std::vector<std::vector<int>> &obs) {
    if(generator_MIP) {
        learn_batch(obs, *generator_hardEM);
    }
    else {
        std::cout << "ERROR: In learn_MIP generator_MIP or constraint_oracle must be defined." << std::endl;
    }
}
void CHMM::learn_MIP(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_MIP(obsVec);
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

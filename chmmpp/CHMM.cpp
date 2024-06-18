#include "CHMM.hpp"
#include "inference/inference.hpp"

namespace chmmpp {

//TODO - add max_iterations
void CHMM::run(const int &T, std::vector<int> &observedStates, std::vector<int> &hiddenStates)
{
    if(!constraint_oracle) {
        std::cout << "Error: To call the function run in CHMM, you must overload it or provide a constraint oracle." << std::endl << std::endl;
        return;
    }
    while(true) {
        hmm.run(T,observedStates,hiddenStates);
        if((*constraint_oracle)(hiddenStates)) {
            break;
        }
    }
}

void CHMM::aStar(const std::vector<int> &observations, std::vector<int> &hidden_states,
                 double &logProb)
{
    if(!constraint_oracle) {
        std::cout << "Error: To call the function aStar in CHMM, you must overload it or provide a constraint oracle." << std::endl << std::endl;
        return;
    }
    std::vector<std::vector<int>> hidden_states_vec;
    std::vector<double> logProb_vec;
    logProb_vec.push_back(logProb);
    chmmpp::aStar_oracle(hmm, observations, hidden_states_vec, logProb_vec, constraint_oracle, 1, this->get_options());
    hidden_states = hidden_states_vec[0];
    logProb = logProb_vec[0];
}

void CHMM::aStarMult(const std::vector<int> &observations,
                     std::vector<std::vector<int>> &hidden_states, std::vector<double> &logProb,
                     const int numSolns)
{
    if(!constraint_oracle) {
        std::cout << "Error: To call the function aStarMult in CHMM, you must overload it or provide a constraint oracle." << std::endl << std::endl;
        return;
    }
    chmmpp::aStar_oracle(hmm, observations, hidden_states, logProb, constraint_oracle, numSolns,
                            this->get_options());
}

void CHMM::mip_map_inference(const std::vector<int> &observations, std::vector<int> &hidden_states,
                             double &logProb)
{
    std::cout << "Warning: mip_map_inference not defined in CHMM." << std::endl << std::endl;
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

void CHMM::learn_stochastic_constraint_oracle(const std::vector<std::vector<int>> &obs) {
    if(constraint_oracle) {
        Generator_Stochastic _generator_stochastic(constraint_oracle);
        learn_batch(obs,_generator_stochastic);
    }
    else {
        std::cout << "ERROR: In learn_stochastic_constraint_oracle constraint_oracle must be defined." << std::endl;
    }
}
void CHMM::learn_stochastic_constraint_oracle(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_stochastic_constraint_oracle(obsVec);
}

void CHMM::learn_stochastic_generator(const std::vector<std::vector<int>> &obs) {
    if(generator_stochastic) {
        learn_batch(obs, *generator_stochastic);
    }
    else {
        std::cout << "ERROR: In learn_stochastic_generator generator_stochastic." << std::endl;
    }
}
void CHMM::learn_stochastic_generator(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_stochastic_generator(obsVec);
}

void CHMM::learn_hardEM_constraint_oracle(const std::vector<std::vector<int>> &obs) {
    if(constraint_oracle) {
        Generator_HardEM _generator_hardEM(constraint_oracle);
        learn_batch(obs,_generator_hardEM);
    }
    else {
        std::cout << "ERROR: In learn_hardEM constraint_oracle must be defined." << std::endl;
    }
}
void CHMM::learn_hardEM_constraint_oracle(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_hardEM_constraint_oracle(obsVec);
}

void CHMM::learn_hardEM_generator(const std::vector<std::vector<int>> &obs) {
    if(generator_stochastic) {
        learn_batch(obs, *generator_stochastic);
    }
    else {
        std::cout << "ERROR: In learn_stochastic generator_stochastic must be defined." << std::endl;
    }
}
void CHMM::learn_hardEM_generator(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_stochastic_generator(obsVec);
}

void CHMM::learn_MIP_generator(const std::vector<std::vector<int>> &obs) {
    if(generator_MIP) {
        learn_batch(obs, *generator_MIP);
    }
    else {
        std::cout << "ERROR: In learn_MIP generator_MIP or constraint_oracle must be defined." << std::endl;
    }
}
void CHMM::learn_MIP_generator(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    learn_MIP_generator(obsVec);
}

double CHMM::log_likelihood_estimate(const std::vector<std::vector<int>> &obs){
    if(constraint_oracle) {
        return chmmpp::log_likelihood_estimate(hmm, constraint_oracle, obs, this->get_options());
    }
    else {
        std::cout << "ERROR: In log_likelihood_estimate, constraint_oracle must be defined." << std::endl;
    }
}
double CHMM::log_likelihood_estimate(const std::vector<int> &obs) {
    std::vector<std::vector<int>> obsVec;
    obsVec.push_back(obs);
    return log_likelihood_estimate(obsVec);
}



void CHMM::learn_semisupervised_hardEM(const std::vector<std::vector<int>> &supervisedObs,
                                       const std::vector<std::vector<int>> &supervisedHidden,
                                       const std::vector<std::vector<int>> &unsupervisedObs)
{
    //TODO
    return;
}

}  // namespace chmmpp

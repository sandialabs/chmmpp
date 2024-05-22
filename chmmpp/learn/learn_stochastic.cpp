#include <iostream>
#include "learn.hpp"

namespace chmmpp {

void process_options(const Options &options, double &convergence_tolerance, unsigned int &C, unsigned int& max_iterations)
{
    for (const auto &it : options.options) {
        if (it.first == "C") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp > 0)
                    C = tmp;
                else
                    std::cerr << "WARNING: 'C' option must be a non-negative integer" << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second)) {
                C = std::get<unsigned int>(it.second);
            }
            else
                std::cerr << "WARNING: 'C' option must be a non-negative integer" << std::endl;
        }
        else if (it.first == "max_iterations") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp > 0)
                    max_iterations = tmp;
                else
                    std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer" << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second)) {
                max_iterations = std::get<unsigned int>(it.second);
            }
            else
                std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer" << std::endl;
        }
        else if (it.first == "convergence_tolerance") {
            if (std::holds_alternative<double>(it.second))
                convergence_tolerance = std::get<double>(it.second);
            else
                std::cerr << "WARNING: 'convergence_tolerance' option must be a double"
                          << std::endl;
        }
    }
}

std::vector<std::vector<std::vector<int>>> generator (
                    HMM &hmm, const int& num_solutions, const int& max_iterations, 
                    const std::vector<std::vector<int>>& obs, 
                    const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle)
{
    std::vector<std::vector<std::vector<int>>> output;

    for(size_t r = 0; r < obs.size(); ++r) {
        std::vector<std::vector<int>> tempHiddenVec;
        while(tempHiddenVec.size() < num_solutions) {
            auto tempHidden = hmm.generateHidden(obs[r]);
            if(constraintOracle[r](tempHidden)) {
                tempHiddenVec.push_back(tempHidden);
            }
        }
        output.push_back(tempHiddenVec);
    }
    return output;
}

// Will work best/fastest if the sets of hidden states which satisfy the constraints
// This algorithm is TERRIBLE, I can't even get it to converge in a simple case with T = 10.
// This is currently the only learning algorithm we have for having a constraint oracle rather than
// ``simple'' constraints This also fails to work if we are converging towards values in the
// transition matrix with 0's (which is NOT uncommon)
void learn_stochastic(HMM &hmm, const std::vector<std::vector<int> > &obs,
                      const std::vector<std::function<bool(std::vector<int> &)> > &constraintOracle,
                      const double convergence_tolerance, unsigned int C, unsigned int max_iterations)
{
    Options options;
    learn_batch(hmm, obs, constraintOracle, generator, options);
}

void learn_stochastic(HMM &hmm, const std::vector<std::vector<int> > &obs,
                      const std::vector<std::function<bool(std::vector<int> &)> > &constraintOracle,
                      const Options &options)
{
    double convergence_tolerance = 10E-6;
    unsigned int C = 10E4;
    unsigned int max_iterations = 1000;
    process_options(options, convergence_tolerance, C, max_iterations);

    learn_stochastic(hmm, obs, constraintOracle, convergence_tolerance, C, max_iterations);
}

void learn_stochastic(HMM &hmm, const std::vector<int> &obs,
                      const std::function<bool(std::vector<int> &)> &constraintOracle,
                      const Options &options)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    std::vector<std::function<bool(std::vector<int> &)> > newConstraintOracle;
    newConstraintOracle.push_back(constraintOracle);
    learn_stochastic(hmm, newObs, newConstraintOracle, options);
}

}  // namespace chmmpp

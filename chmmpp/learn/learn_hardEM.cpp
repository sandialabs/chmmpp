#include <iostream>
#include "learn.hpp"
#include "../inference/inference.hpp"

namespace chmmpp {

namespace {

void process_options(const Options& options, double& convergence_tolerance,
                     unsigned int& max_iterations)
{
    for (const auto& it : options.options) {
        if (it.first == "max_iterations") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp > 0)
                    max_iterations = tmp;
                else
                    std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer"
                              << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second)) {
                max_iterations = std::get<unsigned int>(it.second);
            }
            else
                std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer"
                          << std::endl;
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
                    const HMM &hmm, const int& num_solutions, const int& max_iterations, 
                    const std::vector<std::vector<int>>& obs, 
                    const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle)
{
    std::vector<std::vector<std::vector<int>>> output;

    for(size_t r = 0; r < obs.size(); ++r) {
        std::vector<std::vector<int>> tempHidden;
        std::vector<double> temp;
        aStarMultOracle(hmm, obs[r], tempHidden, temp, constraintOracle[r], num_solutions,
                            max_iterations);
        output.push_back(tempHidden);
    }
    return output;
}

}  // namespace



void learn_hardEM(HMM& hmm, const std::vector<std::vector<int> >& obs,
                  const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle,
                  const int numSolns, const Options& options)
{
    learn_batch(hmm, obs, constraintOracle, generator, options);
}

void learn_hardEM(HMM& hmm, const std::vector<int>& obs,
                  const std::function<bool(std::vector<int>&)>& constraintOracle,
                  const int numSolns, const Options& options)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    std::vector<std::function<bool(std::vector<int>&)> > newConstraintOracle;
    newConstraintOracle.push_back(constraintOracle);
    // WEH - Is this an error???
    // learn_stochastic(hmm, newObs, newConstraintOracle, numSolns, options);
    learn_hardEM(hmm, newObs, newConstraintOracle, numSolns, options);
}

}  // namespace chmmpp

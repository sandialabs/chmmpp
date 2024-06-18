#pragma once

#include <functional>
#include <vector>
#include <chmmpp/HMM.hpp>
#include <chmmpp/inference/inference.hpp>
#include <chmmpp/Constraint_Oracle.hpp>

namespace chmmpp {

class Generator_Base {
public:
    int num_solutions = 1;
    int max_iterations = 1000000; //TODO implement
    virtual std::vector<std::vector<std::vector<int>>> operator() (
        HMM &hmm, const std::vector<std::vector<int>>& obs
    ) const = 0; //[obs.size()][num_solutions][t]
};

class Generator_Stochastic : public Generator_Base {
public:
    std::shared_ptr<Constraint_Oracle_Base> constraint_oracle;

    virtual std::vector<std::vector<std::vector<int>>> operator()(
        HMM &hmm, const std::vector<std::vector<int>>& obs
    ) const
    {
        std::vector<std::vector<std::vector<int>>> output;

        for(size_t r = 0; r < obs.size(); ++r) {
            std::vector<std::vector<int>> tempHiddenVec;
            while(tempHiddenVec.size() < num_solutions) {
                auto tempHidden = hmm.generateHidden(obs[r]);
                if((*constraint_oracle)(tempHidden)) {
                    tempHiddenVec.push_back(tempHidden);
                }
            }
            output.push_back(tempHiddenVec);
        }
        return output;
    }

    Generator_Stochastic(std::shared_ptr<Constraint_Oracle_Base> _constraint_oracle) {
        constraint_oracle = _constraint_oracle;
    }
};

class Generator_HardEM : public Generator_Base {
public:
    std::shared_ptr<Constraint_Oracle_Base> constraint_oracle;

    virtual std::vector<std::vector<std::vector<int>>> operator()(
        HMM &hmm, const std::vector<std::vector<int>>& obs
    ) const
    {
        std::vector<std::vector<std::vector<int>>> output;

        for(size_t r = 0; r < obs.size(); ++r) {
            std::vector<std::vector<int>> tempHidden;
            std::vector<double> temp;
            aStar_oracle(hmm, obs[r], tempHidden, temp, constraint_oracle, num_solutions,
                                max_iterations);
            output.push_back(tempHidden);
        }
        return output;
    }

    Generator_HardEM(std::shared_ptr<Constraint_Oracle_Base> _constraint_oracle) {
        constraint_oracle = _constraint_oracle;
    }
};


void estimate_hmm(HMM &hmm, const std::vector<std::vector<int> > &obs,
                  const std::vector<std::vector<int> > &hid);

void learn_unconstrained(HMM &hmm, const std::vector<int> &obs);
void learn_unconstrained(HMM &hmm, const std::vector<std::vector<int> > &obs);

void learn_batch(HMM &hmm, 
                const std::vector<std::vector<int>>& obs, 
                const Generator_Base& generator,
                const Options& options); 

void learn_semisupervised_hardEM(HMM &hmm, const std::vector<std::vector<int> > &supervisedObs,
                                 const std::vector<std::vector<int> > &supervisedHidden,
                                 const std::vector<std::vector<int> > &unsupervisedObs,
                                 const std::function<bool(std::vector<int> &)> &constraintOracle,
                                 bool partialOracle, const Options &options);
                                
double log_likelihood_estimate(HMM &hmm,
                           std::shared_ptr<Constraint_Oracle_Base> &constraint_oracle,
                           const std::vector<std::vector<int>> &obs, 
                           const Options& options); 

}  // namespace chmmpp

#include <iostream>
#include "learn.hpp"
#include "../inference/inference.hpp"
#include <iomanip>


namespace chmmpp {


double log_likelihood_estimate(HMM &hmm,
                           std::shared_ptr<Constraint_Oracle_Base> &constraint_oracle,
                           const std::vector<std::vector<int>> &obs, 
                           const Options& options) 
{
    //TODO Make into options
    size_t num_it = 100000;
    size_t T = obs[0].size();
    for(size_t r = 1; r < obs.size(); ++r) {
        if(obs[r].size() != T) {
            std::cout << "ERROR: Likelihood estimate does not currently working with observations of different sizes." << std::endl;
            return -1;
        }
    }

    std::vector<double> probs(obs.size());

    size_t n = 0;
    while(n < num_it) {
        std::vector<int> hid;
        std::vector<int> tempObs;
        hmm.run(T, tempObs, hid); 
        ++n;
        if((*constraint_oracle)(hid)) {
            for(size_t r = 0; r < obs.size(); ++r) {
                probs[r] += exp(hmm.logProb(obs[r], hid));
            }
        }
    }

    double output = 0.;

    for(size_t r = 0; r < obs.size(); ++r) {
        output += log(probs[r]/((double)n));
    }

    return output;
}

} //namespace chmppp
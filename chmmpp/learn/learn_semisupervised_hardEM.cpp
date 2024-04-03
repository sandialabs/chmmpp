#include <iostream>
#include "learn.hpp"

namespace chmmpp {

void process_options(const Options &options, double &convergence_tolerance, unsigned int &C, double &gamma)
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
        else if (it.first == "convergence_tolerance") {
            if (std::holds_alternative<double>(it.second))
                convergence_tolerance = std::get<double>(it.second);
            else
                std::cerr << "WARNING: 'convergence_tolerance' option must be a double"
                          << std::endl;
        }
    }
}

// Will work best/fastest if the sets of hidden states which satisfy the constraints
// This algorithm is TERRIBLE, I can't even get it to converge in a simple case with T = 10.
// This is currently the only learning algorithm we have for having a constraint oracle rather than
// ``simple'' constraints This also fails to work if we are converging towards values in the
// transition matrix with 0's (which is NOT uncommon)
void learn_semisupervised_hardEM(HMM &hmm, const std::vector< std::vector<int> > &supervisedObs, 
                           const std::vector< std::vector<int> > &supervisedHidden,
                           const std::vector< std::vector<int> > &unsupervisedObs,
                           const Options &options) 
{    
    std::cout << "TEST!" << std::endl;
}

}  // namespace chmmpp

#include <iostream>
#include "citationHMM.hpp"

namespace chmmpp {

void citationHMM::learn_citation_semisupervised_hard(const std::vector< std::vector<std::string> > &supervisedWords, const std::vector< std::vector<std::string> > &supervisedCategories, const std::vector< std::vector<std::string> > &unsupervisedWords)
{
    auto option = get_option<double>("convergence_tolerance");
    double convergence_tolerance = 10E-6;
    if (option.has_value())
        convergence_tolerance = *option;
    std::vector<std::vector<int> > supervisedObs;
    std::vector<std::vector<int> > supervisedHidden;
    std::vector<std::vector<int> > unsupervisedObs;
    this->learn_semisupervised_hardEM(supervisedObs, supervisedHidden, unsupervisedObs);
}

}  // namespace chmmpp

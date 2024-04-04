#include <iostream>
#include "citationHMM.hpp"

namespace chmmpp {

void citationHMM::learn_citation_semisupervised_hard(const std::vector< std::vector<std::string> > &supervisedWords, const std::vector< std::vector<std::string> > &supervisedCategories, const std::vector< std::vector<std::string> > &unsupervisedWords)
{
    //CLM - TODO -- I don't really understand how option works right now
    auto option = get_option<double>("convergence_tolerance");
    double convergence_tolerance = 10E-6;
    if (option.has_value())
        convergence_tolerance = *option;
    
    //Convert from strings to ints using our maps
    std::vector<std::vector<int> > supervisedObs;
    std::vector<std::vector<int> > supervisedHidden;
    std::vector<std::vector<int> > unsupervisedObs;

    int index = 0;
    for(const auto &line: supervisedWords) {
        supervisedObs.push_back({});
        for(const auto &word: line) {
            supervisedObs[index].push_back(wordMap[word]);
        }
        ++index;
    }

    index = 0;
    for(const auto &line: supervisedCategories) {
        supervisedHidden.push_back({});
        for(const auto &word: line) {
            supervisedHidden[index].push_back(categoryMap[word]);
        }
        ++index;
    }

    index = 0;
    for(const auto &line: unsupervisedWords) {
        unsupervisedObs.push_back({});
        for(const auto &word: line) {
            if(wordMap.count(word) != 0) {
                unsupervisedObs[index].push_back(wordMap[word]);
            }
            else {
                unsupervisedObs[index].push_back(0);
            }

        }
        ++index;
    }

    this->learn_semisupervised_hardEM(supervisedObs, supervisedHidden, unsupervisedObs);
}

}  // namespace chmmpp

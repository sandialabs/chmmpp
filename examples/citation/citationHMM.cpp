#include "citationHMM.hpp"

namespace chmmpp {

//Make sure to initalize with supervised data
citationHMM::citationHMM(const std::vector< std::vector<std::string> > &supervisedWords, const std::vector< std::vector<std::string> > &supervisedCategories)
{
    //Allows us to map words and categories to hidden states
    wordMap.clear();
    categoryMap.clear();
    
    int counter = 0;
    for(const auto &line :supervisedWords) {
        for(const auto &word : line) {
            if(wordMap.count(word) == 0) {
                wordMap[word] = counter;
                ++counter;
            }
        }
    }

    counter = 0;
    for(const auto &line :supervisedCategories) {
        for(const auto &word : line) {
            if(categoryMap.count(word) == 0) {
                categoryMap[word] = counter;
                ++counter;
            }
        }
    }
}

}  // namespace chmmpp

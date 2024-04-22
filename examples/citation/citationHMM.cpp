#include "citationHMM.hpp"
#include <iostream>

namespace chmmpp {

// Make sure to initalize with supervised data
citationHMM::citationHMM(const std::vector<std::vector<std::string> > &supervisedWords,
                         const std::vector<std::vector<std::string> > &supervisedCategories)
{
    // Allows us to map words and categories to hidden states
    // wordMAP IS 1 INDEXED!
    // This allows us to have the unknown state always be 0
    wordMap.clear();
    categoryMap.clear();

    int counter = 1;
    for (const auto &line : supervisedWords) {
        for (const auto &word : line) {
            if (wordMap.count(word) == 0) {
                wordMap[word] = counter;
                ++counter;
            }
        }
    }

    counter = 0;
    for (const auto &line : supervisedCategories) {
        for (const auto &word : line) {
            if (categoryMap.count(word) == 0) {
                categoryMap[word] = counter;
                ++counter;
            }
        }
    }

    // Set HMM
    int O = wordMap.size() + 1;  // Allows for UNKNOWN state
    int H = categoryMap.size();

    std::vector<std::vector<double> > _A(H);
    std::vector<double> _S(H, 1. / H);
    std::vector<std::vector<double> > _E(H);

    for (auto &vec : _A) {
        vec.resize(H, 1. / H);
    }

    for (auto &vec : _E) {
        vec.resize(O, 1. / O);
    }

    this->initialize(_A, _S, _E);

    // Constraint Oracle
    // Categories appear in chunks and only at most once
    constraintOracle = [](std::vector<int> &hid) -> bool {
        for (int i = 2; i < hid.size(); ++i) {
            if (hid[i] != hid[i - 1]) {
                for (int j = 0; j < i - 1; ++j) {
                    if (hid[i] == hid[j]) {
                        return false;
                    }
                }
            }
        }

        return true;
    };

    this->partialOracle = true;  // Only needs partial sequences
}

}  // namespace chmmpp

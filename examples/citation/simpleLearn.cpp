//
// Learn HMM parameters
//
// Generating random trials where the number of nonzeros is fixed
//
#include <iostream>
#include <fstream>
#include "citationHMM.hpp"

int main()
{
    std::ifstream supervisedFile;
    supervisedFile.open("/Users/clmatte/Desktop/HMM/CHMMPP/chmmpp/examples/citation/data/supervised.txt"); //TODO: FIX when we decide what to do with data
    if(!supervisedFile.is_open()) {
        std::cout << "AHHHHHH" << std::endl;
        return -1;
    }
    std::vector< std::vector<std::string> > supervisedWords;
    std::vector< std::vector<std::string> > supervisedCategories;
    chmmpp::readFile(supervisedFile, supervisedWords, supervisedCategories);

    for(const auto &line: supervisedCategories) {
        for(const auto &word: line) {
            std::cout << word << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

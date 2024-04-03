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
    //Read in data from files
    std::ifstream supervisedFile("/Users/clmatte/Desktop/HMM/CHMMPP/chmmpp/examples/citation/data/supervised.txt"); //TODO: FIX when we decide what to do with data
    std::ifstream unsupervisedFile("/Users/clmatte/Desktop/HMM/CHMMPP/chmmpp/examples/citation/data/unsupervised.txt");
    std::ifstream testFile("/Users/clmatte/Desktop/HMM/CHMMPP/chmmpp/examples/citation/data/test.txt");

    std::vector< std::vector<std::string> > supervisedWords;
    std::vector< std::vector<std::string> > supervisedCategories;
    chmmpp::readFile(supervisedFile, supervisedWords, supervisedCategories);

    std::vector< std::vector<std::string> > unsupervisedWords;
    std::vector< std::vector<std::string> > unsupervisedCategories;
    chmmpp::readFile(supervisedFile, unsupervisedWords, unsupervisedCategories);

    std::vector< std::vector<std::string> > testWords;
    std::vector< std::vector<std::string> > testCategories;
    chmmpp::readFile(supervisedFile, testWords, testCategories);

    chmmpp::citationHMM hmm(supervisedWords, supervisedCategories); //Initialize HMM

    for(const auto &myPair: hmm.categoryMap) {
        std::cout << myPair.first << " " << myPair.second << "\n";
    }

    return 0;
}

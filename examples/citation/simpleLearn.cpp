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
    // Read in data from files
    std::ifstream supervisedFile(
        "/Users/clmatte/Desktop/HMM/CHMMPP/chmmpp/examples/citation/data/supervised.txt");  // TODO:
                                                                                            // FIX
                                                                                            // when
                                                                                            // we
                                                                                            // decide
                                                                                            // what
                                                                                            // to do
                                                                                            // with
                                                                                            // data
    std::ifstream unsupervisedFile(
        "/Users/clmatte/Desktop/HMM/CHMMPP/chmmpp/examples/citation/data/unsupervised.txt");
    std::ifstream testFile(
        "/Users/clmatte/Desktop/HMM/CHMMPP/chmmpp/examples/citation/data/test.txt");

    std::vector<std::vector<std::string> > supervisedWords;
    std::vector<std::vector<std::string> > supervisedCategories;
    chmmpp::readFile(supervisedFile, supervisedWords, supervisedCategories);

    std::vector<std::vector<std::string> > unsupervisedWords;
    std::vector<std::vector<std::string> > unsupervisedCategories;
    chmmpp::readFile(unsupervisedFile, unsupervisedWords, unsupervisedCategories);

    std::vector<std::vector<std::string> > testWords;
    std::vector<std::vector<std::string> > testCategories;
    chmmpp::readFile(testFile, testWords, testCategories);

    chmmpp::citationHMM chmm(supervisedWords, supervisedCategories);  // Initialize HMM

    chmm.learn_citation_semisupervised_hard(supervisedWords, supervisedCategories,
                                            unsupervisedWords);

    return 0;
}

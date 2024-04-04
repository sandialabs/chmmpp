#pragma once

#include <chmmpp/chmmpp.hpp>
#include <unordered_map>

namespace chmmpp {

//We assume that we may have unknown words (wordMap[UNKNOWN] = 0), but not unknown categories
class citationHMM : public CHMM {
   public:
    std::unordered_map<std::string, int> categoryMap;
    std::unordered_map<std::string, int> wordMap;

   public:
    citationHMM(const std::vector< std::vector<std::string> > &supervisedWords, const std::vector< std::vector<std::string> > &supervisedCategories);

    // A tailored aStar implementation
    void aStar_citation(const std::vector<int> &observations, std::vector<int> &hidden_states,
                        double &logProb);

    // A tailored aStar implementation
    void aStarMult_citation(const std::vector<int> &observations,
                            std::vector<std::vector<int>> &hidden_states,
                            std::vector<double> &logProb, int numSolns);

    // Optimize using an mixed-integer programming formulation
    void mip_map_inference_citation(const std::vector<int> &observations, std::vector<int> &hidden_states,
                           double &logProb);

    void learn_citation_semisupervised_hard(const std::vector< std::vector<std::string> > &supervisedWords, const std::vector< std::vector<std::string> > &supervisedCategories, const std::vector< std::vector<std::string> > &unsupervisedWords);
};

void readFile(std::ifstream &inputFile, std::vector< std::vector<std::string> > &words, std::vector< std::vector<std::string> > &categories);  

}  // namespace chmmpp

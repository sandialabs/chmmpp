#pragma once

#include <chmmpp/chmmpp.hpp>

namespace chmmpp {

//
// Constrained HMM inference where the number of times the hidden state 'zero' is visited is fixed.
//
class citationHMM : public CHMM {
   public:
    int numZeros;

   public:
    citationHMM(int _numZeros);

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

    void learn_citation(const std::vector<std::vector<int>> &observations);
    void learn_citation(const std::vector<int> &observations);
};

void readFile(std::ifstream &inputFile, std::vector< std::vector<std::string> > &words, std::vector< std::vector<std::string> > &categories);

}  // namespace chmmpp

#pragma once

#include <chmmpp/chmmpp.hpp>

namespace chmmpp {

//
// Constrained HMM inference where the number of times the hidden state 'zero' is visited is fixed.
//
class numZerosHMM : public CHMM {
   public:
    int numZeros;

   public:
    numZerosHMM(int _numZeros);

    // A tailored aStar implementation
    void aStar_numZeros(const std::vector<int> &observations, std::vector<int> &hidden_states,
                        double &logProb);

    // A tailored aStar implementation
    void aStarMult_numZeros(const std::vector<int> &observations,
                            std::vector<std::vector<int>> &hidden_states,
                            std::vector<double> &logProb, int numSolns);

    // Optimize using an mixed-integer programming formulation
    void mip_map_inference(const std::vector<int> &observations, std::vector<int> &hidden_states,
                           double &logProb);
};

}  // namespace chmmpp

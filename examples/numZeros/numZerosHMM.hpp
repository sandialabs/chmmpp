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

    //
    // Inference methods
    //
    // TODO - Consistent naming scheme for inference
    //

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

    //
    // Learning methods
    //
    // TODO - Consistent naming scheme

    // Customized Soft EM?
    void learn_numZeros(const std::vector<std::vector<int>> &observations);
    void learn_numZeros(const std::vector<int> &observations);

    // SAEM using MIP to generate samples
    void learn_mip(const std::vector<std::vector<int>> &observations);
    void learn_mip(const std::vector<int> &observations)
    {
        std::vector<std::vector<int>> tmp;
        tmp.push_back(observations);
        learn_mip(tmp);
    }
};

}  // namespace chmmpp

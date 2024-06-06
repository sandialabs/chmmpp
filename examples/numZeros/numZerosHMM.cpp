#include "numZerosHMM.hpp"
#include "numZeros_mip.hpp"

namespace chmmpp {

class Constraint_Oracle_NumZeros : public Constraint_Oracle_Base {
public: 
    int numZeros;
    bool operator()(std::vector<int> hid) {
        size_t count = 0;
        for(const auto& val: hid) {
            if(!val) {
                ++count;
            }
        }
        return (count == numZeros);
    }

    Constraint_Oracle_NumZeros(int _numZeros) {
        numZeros = _numZeros;
    }
};

void project(const HMM& hmm, const std::vector<int> &obs, std::vector<int> &hidden, const int &num_zeros) {
    LearningModel model;
    numZerosHMM nzhmm(num_zeros);
    nzhmm.initialize(hmm);
    model.set_options(nzhmm.get_options());
    model.initialize(nzhmm, obs, hidden);
    double log_likelihood;
    model.optimize(log_likelihood, hidden);
}

class Generator_MIP_NumZeros : public Generator_Base {
   public:
    size_t numZeros;

    virtual std::vector<std::vector<std::vector<int>>> operator()(
        HMM &hmm, const std::vector<std::vector<int>>& obs
    ) const
    {
        std::vector<std::vector<std::vector<int>>> output(obs.size());
        for(size_t r = 0; r < obs.size(); ++r) {
            for(size_t b = 0; b < num_solutions; ++b) {
                auto unconstrained_hidden = hmm.generateHidden(obs[r]);
                #ifdef WITH_COEK
                    //project(hmm, obs[r], unconstrained_hidden, numZeros);
                #endif
                output[r].push_back(unconstrained_hidden);
            }
        }
        
        return output;
    }

    Generator_MIP_NumZeros(const size_t& _numZeros) {
        numZeros = _numZeros;
    }
};

numZerosHMM::numZerosHMM(int _numZeros) : numZeros(_numZeros)
{
    constraint_oracle = std::make_shared<Constraint_Oracle_NumZeros>(_numZeros);
    generator_MIP = std::make_shared<Generator_MIP_NumZeros>(_numZeros);
}

}  // namespace chmmpp

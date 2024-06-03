#include "numZerosHMM.hpp"

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

numZerosHMM::numZerosHMM(int _numZeros) : numZeros(_numZeros)
{
    constraint_oracle = std::make_shared<Constraint_Oracle_NumZeros>(_numZeros);
}

}  // namespace chmmpp

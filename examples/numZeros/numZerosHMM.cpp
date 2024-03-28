#include "numZerosHMM.hpp"

namespace chmmpp {

numZerosHMM::numZerosHMM(int _numZeros) 
    : numZeros(_numZeros)
{
constraintOracle = [this](std::vector<int>& hid) -> bool { return this->numZeros == count(hid.begin(), hid.end(), 0); };
}

} // namespace chmmpp

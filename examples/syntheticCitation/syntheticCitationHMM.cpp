#include "syntheticCitationHMM.hpp"

namespace chmmpp {

syntheticCitationHMM::syntheticCitationHMM() 
{
    constraintOracle = [this](std::vector<int>& hid) -> bool {
        for(size_t t1 = 1; t1 < hid.size(); ++t1) {
            if(hid[t1] != hid[t1-1]) {
                for(size_t t2 = 0; t2 < t1-1; ++t2) {
                    if(hid[t1] == hid[t2]) {
                        return false;
                    }
                }
            }
        }
        return true;
    };
}

}  // namespace chmmpp

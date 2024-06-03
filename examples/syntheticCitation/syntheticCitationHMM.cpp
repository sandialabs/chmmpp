#include "syntheticCitationHMM.hpp"

namespace chmmpp {

class Constraint_Oracle_Synthetic_Citation : public Constraint_Oracle_Base {
public: 
    bool operator()(std::vector<int> hid) {
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
    }
};

syntheticCitationHMM::syntheticCitationHMM() 
{
    constraint_oracle = std::make_shared<Constraint_Oracle_Synthetic_Citation>();
}

}  // namespace chmmpp

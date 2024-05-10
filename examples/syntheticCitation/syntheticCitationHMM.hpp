#pragma once

#include <chmmpp/chmmpp.hpp>

namespace chmmpp {

//
// Constrained HMM where hidden states come in blocks that don't repeat
//
class syntheticCitationHMM : public CHMM {

   public:
    syntheticCitationHMM();
};

}  // namespace chmmpp

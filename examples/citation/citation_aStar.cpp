
#include <queue>
#include <iostream>
#include <chmmpp/util/vectorhash.hpp>
#include "citationHMM.hpp"

namespace chmmpp {

//---------------------------------
//-----A star with constraints-----
//---------------------------------

// The same as the function above, however here we are allowed to specify the number of times the
// function is in hidden state 0 with the parameter numZeros Could also expand this to be general
// linear constraints
void citationHMM::aStar_citation(const std::vector<int>& observations,
                                 std::vector<int>& hidden_states, double& logProb)
{
    return;
}

//---------------------
//-----A* multiple-----
//---------------------

// Returns the top numSolns solutions to the inference problem.
// Uses the same inference technique as A*Oracle, so it is much slower than general A*
void citationHMM::aStarMult_citation(const std::vector<int>& observations,
                                     std::vector<std::vector<int>>& hidden_states,
                                     std::vector<double>& logProb, int numSolns)
{
    return;
}

}  // namespace chmmpp

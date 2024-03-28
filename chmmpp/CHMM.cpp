#include "CHMM.hpp"
#include "inference/inference.hpp"

namespace chmmpp {

void CHMM::aStar(const std::vector<int> &observations, std::vector<int> &hidden_states, double &logProb)
{
chmmpp::aStarOracle(hmm, observations, hidden_states, logProb, constraintOracle);
}

void CHMM::aStarMult(const std::vector<int> &observations, std::vector<std::vector<int>> &hidden_states, std::vector<double> &logProb,
                     const int numSolns)
{
chmmpp::aStarMultOracle(hmm, observations, hidden_states, logProb, constraintOracle, numSolns);
}

void CHMM::mip_map_inference(const std::vector<int> &observations, std::vector<int> &hidden_states, double &logProb)
{
hidden_states.resize(hmm.getH());
logProb = 0;
}

} // namespace chmmpp

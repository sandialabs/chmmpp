#include "inference.hpp"
#ifdef WITH_COEK
#    include "LPModel.hpp"
#endif

namespace chmmpp {

// Does inference with a given set of observations
// logProb is the log of the probability that the given states occur (we use logs as otherwise we
// could get numerical underflow) Uses the A* algorithm for inference Without constraints (such as
// in this case) this is basically equivalent to running Viterbi with a bit of overhead
void lp_map_inference(const HMM &hmm, const std::vector<int> &observations,
                      std::vector<int> &hidden_states, double &logProb, const Options &options)
{
#ifdef WITH_COEK
    LPModel model;

    model.set_options(options);
    model.initialize(hmm, observations);
    model.optimize(logProb, hidden_states);
#else
    hidden_states.resize(hmm.getH());
    logProb = 0;
#endif
}

}  // namespace chmmpp

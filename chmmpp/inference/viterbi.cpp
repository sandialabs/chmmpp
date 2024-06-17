// HMM.cpp

#include <iostream>
#include "inference.hpp"
#include "../util/vectorhash.hpp"

namespace chmmpp {

//-----------------
//-----Viterbi-----
//-----------------

// Does inference with a given set of observations
// logProb is the log of the probability that the given states occur (we use logs as otherwise we
// could get numerical underflow) Classic Viterbi algorithm
void viterbi(const HMM& hmm, const std::vector<int>& observations, std::vector<int>& hidden_states,
             double& logProb)
{
    const int T = observations.size();
    auto H = hmm.getH();
    auto O = hmm.getO();
    const auto& A = hmm.getA();
    const auto& S = hmm.getS();
    const auto& E = hmm.getE();

    // So we don't need to keep recomputing logs
    std::vector<std::vector<double> > logA(H);
    std::vector<std::vector<double> > logE(H);

    for (size_t h1 = 0; h1 < H; ++h1) {
        logA[h1].resize(H);
        for (size_t h2 = 0; h2 < H; ++h2) {
            logA[h1][h2] = std::log(A[h1][h2]);
        }
    }

    for (size_t h = 0; h < H; ++h) {
        logE[h].resize(O);
        for (size_t o = 0; o < O; ++o) {
            logE[h][o] = std::log(E[h][o]);
        }
    }

    std::vector<std::vector<double> > v(T);     // V is for Viterbi
    std::vector<std::vector<double> > next(T);  // Used to reconstruct the path

    for (auto& vec : v) vec.resize(H);

    for (auto& vec : next) vec.resize(H);

    for (int t = T - 2; t >= 0; --t) {
        for (size_t h1 = 0; h1 < H; ++h1) {
            v[t][h1] = -10E12;
            for (size_t h2 = 0; h2 < H; ++h2) {
                auto temp = v[t + 1][h2] + logA[h1][h2] + logE[h2][observations[t + 1]];
                if (temp > v[t][h1]) {
                    v[t][h1] = temp;
                    next[t][h1] = h2;
                }
            }
        }
    }

    for (size_t h = 0; h < H; ++h)  // Starting probs
        v[0][h] += std::log(S[h]) + logE[h][observations[0]];

    logProb = -1E12;
    hidden_states.resize(T);
    for (size_t h = 0; h < H; ++h) {
        if (v[0][h] > logProb) {
            hidden_states[0] = h;
            logProb = -v[0][h];
        }
    }

    for (size_t t = 0; t < T - 1; ++t) hidden_states[t + 1] = next[t][hidden_states[t]];

    logProb = -logProb;
    return;
}

}  // namespace chmmpp

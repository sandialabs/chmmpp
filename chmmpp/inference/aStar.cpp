// HMM.cpp

#include <queue>
// #include <iostream>
#include "inference.hpp"
#include "../util/vectorhash.hpp"

namespace chmmpp {

//----------------
//-----A star-----
//----------------

// Does inference with a given set of observations
// logProb is the log of the probability that the given states occur (we use logs as otherwise we
// could get numerical underflow) Uses the A* algorithm for inference Without constraints (such as
// in this case) this is basically equivalent to running Viterbi with a bit of overhead
void aStar(const HMM& hmm, const std::vector<int>& observations, std::vector<int>& hidden_states,
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
    std::vector<double> logS;
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

    // Stands for Viterbi, used as an estimate for how much logprob is left
    std::vector<std::vector<double> > v(T);
    for (auto& vec : v) vec.resize(H);

    /* WEH: Vectors are default initialized to zero.
    for (size_t h = 0; h < H; ++h) {
        v[T - 1][h] = 0;
    }
    */

    for (int t = T - 2; t >= 0; --t) {
        for (size_t h1 = 0; h1 < H; ++h1) {
            double temp = -10E12;
            for (size_t h2 = 0; h2 < H; ++h2) {
                temp = std::max(temp, v[t + 1][h2] + logA[h1][h2] + logE[h2][observations[t + 1]]);
            }
            v[t][h1] = temp;
        }
    }

    std::priority_queue<std::pair<double, std::vector<int> > > openSet;
    std::unordered_map<std::vector<int>, double, vectorHash<int> > gScore;  // log prob so far

    for (int h = 0; h < H; ++h) {
        double tempGScore
            = std::log(S[h]) + logE[h][observations[0]];  // Avoids extra look-up operation
        gScore[{h}] = tempGScore;
        std::vector<int> tempVec = {h};  // make_pair doesn't like taking in {h}
        openSet.push(std::make_pair(tempGScore + v[0][h], tempVec));
    }

    // Actual run run of A* algorithm
    while (!openSet.empty()) {
        auto seq = openSet.top().second;
        openSet.pop();

        int t = seq.size();
        int h1 = seq[t - 1];
        double oldGScore = gScore[seq];

        if (t == T) {
            logProb = oldGScore;
            hidden_states = seq;
            return;
        }

        for (size_t h2 = 0; h2 < H; ++h2) {
            auto newSeq = seq;
            newSeq.push_back(h2);
            double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
            gScore[newSeq] = tempGScore;
            openSet.push(std::make_pair(tempGScore + v[t][h2], newSeq));
        }
    }

    // WEH - What does this mean?  An error?
    hidden_states = {};
    return;
}

}  // namespace chmmpp

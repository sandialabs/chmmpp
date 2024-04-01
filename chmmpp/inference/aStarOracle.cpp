// HMM.cpp

#include <queue>
#include <iostream>
#include "inference.hpp"
#include "../util/vectorhash.hpp"

namespace chmmpp {

//------------------------
//-----A* with oracle-----
//------------------------

// Rather than having some nice function that we can take advantage of the structure of, we just
// have an oracle which Keeps track of all values not just constraint value. Need for more
// complicated constraints Note: This may produce a different solution from other A* functions.
// However, they will have the same logProb, and thus occur with the same probability Effectively
// the same as the code above, but we can't restrict the space if we have too many 0's
void aStarOracle(const HMM& hmm, const std::vector<int>& observations, std::vector<int>& hidden,
                 double& logProb, const std::function<bool(std::vector<int>&)>& constraintOracle)
{
    const int T = observations.size();
    auto H = hmm.getH();
    auto O = hmm.getO();
    const auto& A = hmm.getA();
    const auto& S = hmm.getS();
    const auto& E = hmm.getE();

    // So we don't need to keep recomputing logs
    std::vector<std::vector<double> > logA;
    std::vector<double> logS;
    std::vector<std::vector<double> > logE;

    logA.resize(H);
    logE.resize(H);

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

    std::vector<std::vector<double> > v;  // Stands for Viterbi
    v.resize(T);
    for (int t = 0; t < T; ++t) {
        v[t].resize(H);
    }

    for (size_t h = 0; h < H; ++h) {
        v[T - 1][h] = 0;
    }

    for (int t = T - 2; t >= 0; --t) {
        for (size_t h1 = 0; h1 < H; ++h1) {
            double temp = -10E12;
            for (size_t h2 = 0; h2 < H; ++h2) {
                temp = std::max(temp, v[t + 1][h2] + logA[h1][h2] + logE[h2][observations[t]]);
            }
            v[t][h1] = temp;
        }
    }

    std::vector<int> output;

    // Dist, current h, time, constraint val
    std::priority_queue<std::pair<double, std::vector<int> > >
        openSet;  // Works b/c c++ orders tuples lexigraphically
    std::unordered_map<std::vector<int>, double, boost::hash<std::vector<int> > >
        gScore;  // pair is h,t, constraintVal
    // TODO make better hash for tuple
    // Would gScore be better as a multi-dimensional array? <- probably not, b/c we are hoping it
    // stays sparse
    for (size_t h = 0; h < H; ++h) {
        double tempGScore
            = std::log(S[h]) + logE[h][observations[0]];  // Avoids extra look-up operation

        if (h == 0) {
            std::vector<int> tempVec = {0};  // Otherwise C++ can't figure out what is happening
            openSet.push(std::make_pair(tempGScore + v[0][h], tempVec));
            gScore[{0}] = tempGScore;
        }
        else {
            std::vector<int> tempVec = {1};
            openSet.push(std::make_pair(tempGScore + v[0][h], tempVec));
            gScore[{1}] = tempGScore;
        }
    }

    while (!openSet.empty()) {
        auto tempPair = openSet.top();
        std::vector<int> currentSequence = std::get<1>(tempPair);
        int t = currentSequence.size();
        int h1 = currentSequence[t - 1];

        openSet.pop();
        double oldGScore = gScore.at(currentSequence);
        if (t == T) {
            if (constraintOracle(currentSequence)) {
                logProb = oldGScore;
                hidden = currentSequence;
                return;
            }
        }

        else {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                std::vector<int> newSequence = currentSequence;
                newSequence.push_back(h2);

                gScore[newSequence] = -tempGScore;
                openSet.push(std::make_pair(tempGScore + v[t][h2], newSequence));
            }
        }
    }
}

}  // namespace chmmpp

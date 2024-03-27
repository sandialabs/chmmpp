// HMM.cpp

#include <queue>
#include <iostream>
#include "inference.hpp"
#include "../util/vectorhash.hpp"

namespace chmmpp {

//---------------------------------
//-----A star with constraints-----
//---------------------------------

// The same as the function above, however here we are allowed to specify the number of times the
// function is in hidden state 0 with the parameter numZeros Could also expand this to be general
// linear constraints
void aStar_numZeros(const HMM& hmm, const std::vector<int> &observations, std::vector<int>& hidden_states, double &logProb, const int numZeros)
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

    std::vector<std::vector<double> >
        v;  // Stands for Viterbi, used as an estimate for how much logprob is left
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

    // Dist, current h, time, constraint val
    std::priority_queue<std::tuple<double, int, int, int> >
        openSet;  // Works b/c c++ orders tuples lexigraphically
    std::unordered_map<std::tuple<int, int, int>, double, boost::hash<std::tuple<int, int, int> > >
        gScore;  // pair is h,t, constraintVal
    // TODO make better hash for tuple
    std::unordered_map<std::tuple<int, int, int>, int, boost::hash<std::tuple<int, int, int> > >
        prev;  // Used to recover sequence of hidden states
    for (size_t h = 0; h < H; ++h) {
        double tempGScore
            = std::log(S[h]) + logE[h][observations[0]];  // Avoids extra look-up operation

        if (h == 0) {
            openSet.push(std::make_tuple(tempGScore + v[0][h], 0, 1, 1));
            gScore[std::make_tuple(0, 1, 1)] = tempGScore;
        }
        else {
            openSet.push(std::make_tuple(tempGScore + v[0][h], 1, 1, 0));
            gScore[std::make_tuple(1, 1, 0)] = tempGScore;
        }
    }

    while (!openSet.empty()) {
        auto tempTuple = openSet.top();
        int h1 = std::get<1>(tempTuple);    // Current state
        int t = std::get<2>(tempTuple);     // Current time
        int fVal = std::get<3>(tempTuple);  // Current fVal

        openSet.pop();
        double oldGScore = gScore.at(std::make_tuple(h1, t, fVal));

        if (t == T) {
            if (fVal == numZeros) {  // Make sure we actually satisfy the constraints
                logProb = oldGScore;
                std::vector<int> output;
                output.push_back(h1);

                while (t > 1) {
                    int h = prev[std::make_tuple(h1, t, fVal)];
                    if (h1 == 0) {
                        --fVal;
                    }
                    --t;
                    output.push_back(h);
                    h1 = h;
                }

                std::reverse(output.begin(), output.end());
                hidden_states = output;
                return;
            }
        }

        // Expand in the A* algorithm
        else {
            for (size_t h2 = 0; h2 < H; ++h2) {
                int newFVal = fVal;
                if (h2 == 0) {
                    ++newFVal;
                }

                if (newFVal <= numZeros) {  // Helps reduce the size of the problem - we can't do
                                            // this if we instead have a general oracle
                    double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                    if (gScore.count(std::make_tuple(h2, t + 1, newFVal)) == 0) {
                        gScore[std::make_tuple(h2, t + 1, newFVal)] = tempGScore;
                        openSet.push(std::make_tuple(tempGScore + v[t][h2], h2, t + 1, newFVal));
                        prev[std::make_tuple(h2, t + 1, newFVal)] = h1;
                    }
                    else if (tempGScore > gScore.at(std::make_tuple(
                                 h2, t + 1,
                                 newFVal))) {  // Makes sure we don't have empty call to map
                        gScore.at(std::make_tuple(h2, t + 1, newFVal)) = tempGScore;
                        openSet.push(std::make_tuple(tempGScore + v[t][h2], h2, t + 1, newFVal));
                        prev.at(std::make_tuple(h2, t + 1, newFVal)) = h1;
                    }
                }
            }
        }
    }
}

} // namespace chmmpp

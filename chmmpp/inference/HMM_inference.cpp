// HMM.cpp

#include <queue>
#include <iostream>
#include "HMM_inference.hpp"
#include "../util/vectorhash.hpp"

namespace chmmpp {

//----------------
//-----A star-----
//----------------

// Does inference with a given set of observations
// logProb is the log of the probability that the given states occur (we use logs as otherwise we
// could get numerical underflow) Uses the A* algorithm for inference Without constraints (such as
// in this case) this is basically equivalent to running Viterbi with a bit of overhead
std::vector<int> HMM_inference::aStar(const std::vector<int> &observations, double &logProb) const
{
    const int T = observations.size();

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
            logProb = -oldGScore;
            return seq;
        }

        for (size_t h2 = 0; h2 < H; ++h2) {
            auto newSeq = seq;
            newSeq.push_back(h2);
            double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
            gScore[newSeq] = tempGScore;
            openSet.push(std::make_pair(tempGScore + v[t][h2], newSeq));
        }
    }

    return {};
}

//---------------------------------
//-----A star with constraints-----
//---------------------------------

// The same as the function above, however here we are allowed to specify the number of times the
// function is in hidden state 0 with the parameter numZeros Could also expand this to be general
// linear constraints
std::vector<int> HMM_inference::aStar(const std::vector<int> &observations, double &logProb,
                                      const int numZeros) const
{
    const int T = observations.size();

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
                return output;
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

    return {};
}

//------------------------
//-----A* with oracle-----
//------------------------

// Rather than having some nice function that we can take advantage of the structure of, we just
// have an oracle which Keeps track of all values not just constraint value. Need for more
// complicated constraints Note: This may produce a different solution from other A* functions.
// However, they will have the same logProb, and thus occur with the same probability Effectively
// the same as the code above, but we can't restrict the space if we have too many 0's
std::vector<int> HMM_inference::aStarOracle(
    const std::vector<int> &observations, double &logProb,
    const std::function<bool(std::vector<int>)> &constraintOracle) const
{
    const int T = observations.size();

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
                return currentSequence;
            }
        }

        else {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                std::vector<int> newSequence = currentSequence;
                newSequence.push_back(h2);

                gScore[newSequence] = tempGScore;
                openSet.push(std::make_pair(tempGScore + v[t][h2], newSequence));
            }
        }
    }
    return {};
}

//---------------------
//-----A* multiple-----
//---------------------

// Returns the top numSolns solutions to the inference problem.
// Uses the same inference technique as A*Oracle, so it is much slower than general A*
std::vector<std::vector<int> > HMM_inference::aStarMult(const std::vector<int> &observations,
                                                        double &logProb, const int numZeros,
                                                        const int numSolns) const
{
    const int T = observations.size();

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

    std::vector<std::vector<int> > output;
    int counter = 0;

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
        int fVal = 0;
        for (int i = 0; i < t; ++i) {
            if (currentSequence[i] == 0) {
                ++fVal;
            }
        }

        openSet.pop();
        double oldGScore = gScore.at(currentSequence);
        if (t == T) {
            if (fVal == numZeros) {
                logProb = oldGScore;
                output.push_back(currentSequence);
                ++counter;
                if (counter == numSolns) {
                    return output;
                }
            }
        }

        else {
            for (size_t h2 = 0; h2 < H; ++h2) {
                int newFVal = fVal;
                if (h2 == 0) {
                    ++newFVal;
                }

                if (newFVal <= numZeros) {
                    double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                    std::vector<int> newSequence = currentSequence;
                    newSequence.push_back(h2);

                    gScore[newSequence] = tempGScore;
                    openSet.push(std::make_pair(tempGScore + v[t][h2], newSequence));
                }
            }
        }
    }
    return {};
}

//---------------------------------
//-----A* multiple with Oracle-----
//---------------------------------

// Same as above, but we now have an oracle for the constraints
std::vector<std::vector<int> > HMM_inference::aStarMult(
    const std::vector<int> &observations, double &logProb,
    const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns) const
{
    const int T = observations.size();

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

    // Dist, current h, time, constraint val
    std::priority_queue<std::pair<double, std::vector<int> > >
        openSet;  // Works b/c c++ orders tuples lexigraphically
    std::unordered_map<std::vector<int>, double, boost::hash<std::vector<int> > >
        gScore;  // pair is h,t, constraintVal
    // TODO make better hash for tuple
    // Would gScore be better as a multi-dimensional array? <- probably not, b/c we are hoping it
    // stays sparse
    for (int h = 0; h < H; ++h) {
        double tempGScore
            = std::log(S[h]) + logE[h][observations[0]];  // Avoids extra look-up operation

        std::vector<int> tempVec = {h};  // Otherwise C++ can't figure out what is happening
        openSet.push(std::make_pair(tempGScore + v[0][h], tempVec));
        gScore[tempVec] = tempGScore;
    }

    std::vector<std::vector<int> > output;
    int counter = 0;

    while (!openSet.empty()) {
        auto tempPair = openSet.top();
        openSet.pop();

        std::vector<int> currentSequence = std::get<1>(tempPair);
        int t = currentSequence.size();
        int h1 = currentSequence[t - 1];
        double oldGScore = gScore.at(currentSequence);

        if (t == T) {
            if (constraintOracle(currentSequence)) {
                output.push_back(currentSequence);
                ++counter;
                if (counter == numSolns) {
                    return output;
                }
            }
        }

        else {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double tempGScore = oldGScore + logA[h1][h2] + logE[h2][observations[t]];
                std::vector<int> newSequence = currentSequence;
                newSequence.push_back(h2);

                gScore[newSequence] = tempGScore;
                openSet.push(std::make_pair(tempGScore + v[t][h2], newSequence));
            }
        }
    }
    return output;
}

//-----------------------------------
//-----Learning with constraints-----
//-----------------------------------

// Your HMM should be initalized with your prior guess of the probabilities (referred to as theta in
// the comments) Only for a single set of observations The constraint is the number of zeros in the
// hidden states, denoted by numZeros Epsilon are tolerance This would also work if the constraint
// was a linear combination of the hidden states
void HMM_inference::learn(const std::vector<int> &obs, const int numZeros, const double eps)
{
    int T = obs.size();

    while (true) {
        // alpha
        std::vector<std::vector<std::vector<double> > >
            alpha;  // alpha[c][h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta, c 0's)
        alpha.resize(numZeros + 1);
        for (int c = 0; c <= numZeros; ++c) {
            alpha[c].resize(H);
            for (size_t h = 0; h < H; ++h) {
                alpha[c][h].resize(T, 0.);
                if (((c == 1) && (h == 0)) || ((c == 0) && (h != 0))) {
                    alpha[c][h][0] = S[h] * E[h][obs[0]];
                }
            }
        }

        for (int t = 1; t < T - 1; ++t) {
            for (int c = 0; c <= numZeros; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t h1 = 0; h1 < H; ++h1) {
                        int oldC = c;
                        if (h == 0) {
                            --oldC;
                        }

                        if (oldC >= 0) {
                            alpha[c][h][t] += alpha[oldC][h1][t - 1] * A[h1][h];
                        }
                    }
                    alpha[c][h][t] *= E[h][obs[t]];
                }
            }
        }

        // t = T-1
        for (int c = 0; c <= numZeros; ++c) {
            for (size_t h = 0; h < H; ++h) {
                if (c == numZeros) {
                    for (size_t h1 = 0; h1 < H; ++h1) {
                        int oldC = c;
                        if (h == 0) {
                            --oldC;
                        }

                        if (oldC >= 0) {
                            alpha[c][h][T - 1] += alpha[oldC][h1][T - 2] * A[h1][h];
                        }
                    }
                    alpha[c][h][T - 1] *= E[h][obs[T - 1]];
                }
            }
        }

        // beta
        std::vector<std::vector<std::vector<double> > >
            beta;  // beta[c][h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta, c
                   // 0's )
        beta.resize(numZeros + 1);
        for (int c = 0; c <= numZeros; ++c) {
            beta[c].resize(numZeros + 1);
            for (size_t h = 0; h < H; ++h) {
                beta[c][h].resize(T, 0.);
                if (c == 0) {
                    beta[c][h][T - 1] = 1.;
                }
            }
        }

        for (int t = T - 2; t > 0; --t) {
            for (int c = 0; c <= numZeros; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t h2 = 0; h2 < H; ++h2) {
                        int newC = c;
                        if (h2 == 0) {
                            --newC;
                        }

                        if (newC >= 0) {
                            beta[c][h][t] += beta[newC][h2][t + 1] * A[h][h2] * E[h2][obs[t + 1]];
                        }
                    }
                }
            }
        }

        // t = 0
        // h[0] = 0
        if (numZeros > 0) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                int newC = numZeros - 1;
                if (h2 == 0) {
                    --newC;
                }
                if (newC >= 0) {
                    beta[numZeros - 1][0][0] += beta[newC][h2][1] * A[0][h2] * E[h2][obs[1]];
                }
            }
        }

        // h[0] != 0
        for (size_t h = 1; h < H; ++h) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                int newC = numZeros;
                if (h2 == 0) {
                    --newC;
                }

                if (newC >= 0) {
                    beta[numZeros][h][0] += beta[newC][h2][1] * A[h][h2] * E[h2][obs[1]];
                }
            }
        }

        // den = P(O | theta)
        // Need different denominators because of the scaling
        // This is numerically a VERY weird algorithm
        std::vector<double> den;
        for (int t = 0; t < T; ++t) {
            den.push_back(0.);
            for (size_t h = 0; h < H; ++h) {
                for (int c = 0; c <= numZeros; ++c) {
                    den[t] += alpha[c][h][t] * beta[numZeros - c][h][t];
                }
            }
        }

        // Gamma
        std::vector<std::vector<double> > gamma;  // gamma[h][t] = P(H_t = h | Y , theta)
        gamma.resize(H);
        for (size_t h = 0; h < H; ++h) {
            gamma[h].resize(T);
        }

        for (size_t h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                double num = 0.;
                for (int c = 0; c <= numZeros; ++c) {
                    num += alpha[c][h][t] * beta[numZeros - c][h][t];
                }
                gamma[h][t] = num / den[t];
            }
        }

        // xi
        std::vector<std::vector<std::vector<double> > >
            xi;  // xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta)
        xi.resize(H);
        for (size_t h1 = 0; h1 < H; ++h1) {
            xi[h1].resize(H);
            for (size_t h2 = 0; h2 < H; ++h2) {
                xi[h1][h2].resize(T - 1);
            }
        }

        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                for (int t = 0; t < T - 1; ++t) {
                    double num = 0.;

                    for (int c = 0; c <= numZeros; ++c) {
                        int middleC = 0;
                        if (h2 == 0) {
                            ++middleC;
                        }

                        if (numZeros - middleC - c >= 0) {
                            num += alpha[c][h1][t] * beta[numZeros - middleC - c][h2][t + 1];
                        }
                    }
                    num *= A[h1][h2] * E[h2][obs[t + 1]];

                    xi[h1][h2][t] = num / den[t];
                }
            }
        }

        // New S
        for (size_t h = 0; h < H; ++h) {
            S[h] = gamma[h][0];
        }

        // New E
        for (size_t h = 0; h < H; ++h) {
            for (size_t o = 0; o < O; ++o) {
                double num = 0.;
                double newDen = 0.;

                for (int t = 0; t < T; ++t) {
                    if (obs[t] == o) {
                        num += gamma[h][t];
                    }
                    newDen += gamma[h][t];
                }

                E[h][o] = num / newDen;
            }
        }

        double tol = 0.;

        // New A
        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;

                for (int t = 0; t < T - 1; ++t) {
                    num += xi[h1][h2][t];
                    newDen += gamma[h1][t];
                }
                tol = std::max(std::fabs(A[h1][h2] - num / newDen), tol);
                A[h1][h2] = num / newDen;
            }
        }

        std::cout << "Tolerance: " << tol << "\n";  // Can comment this out if too much printing

        if (tol < eps) {
            break;
        }
    }
}

//----------------------------------------------------------------
//-----Constrained Learning with Multiple Set of Observations-----
//----------------------------------------------------------------

// This is the exact same as the algorithm above, but here we allow multiple observations
// Using the same terminology as the Wikipedia page, we use r to deal with constraints
void HMM_inference::learn(const std::vector<std::vector<int> > &obs,
                          const std::vector<int> &numZeros, const double eps)
{
    int T = obs[0].size();
    size_t R = obs.size();

    while (true) {
        std::vector<std::vector<std::vector<double> > > totalGamma;
        std::vector<std::vector<std::vector<std::vector<double> > > > totalXi;
        for (size_t r = 0; r < R; ++r) {
            // alpha
            std::vector<std::vector<std::vector<double> > >
                alpha;  // alpha[c][h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta, c
                        // 0's)
            alpha.resize(numZeros[r] + 1);
            for (int c = 0; c <= numZeros[r]; ++c) {
                alpha[c].resize(H);
                for (size_t h = 0; h < H; ++h) {
                    alpha[c][h].resize(T);

                    if (((c == 1) && (h == 0)) || ((c == 0) && (h != 0))) {
                        alpha[c][h][0] = S[h] * E[h][obs[r][0]];
                    }

                    else {
                        alpha[c][h][0] = 0.;
                    }
                }
            }

            for (int t = 1; t < T - 1; ++t) {
                for (int c = 0; c <= numZeros[r]; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        alpha[c][h][t] = 0.;
                        for (size_t h1 = 0; h1 < H; ++h1) {
                            int oldC = c;
                            if (h1 == 0) {
                                --oldC;
                            }

                            if (oldC >= 0) {
                                alpha[c][h][t] += alpha[oldC][h1][t - 1] * A[h1][h];
                            }
                        }
                        alpha[c][h][t] *= E[h][obs[r][t]];
                    }
                }
            }

            // t = T-1
            for (int c = 0; c <= numZeros[r]; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    alpha[c][h][T - 1] = 0.;
                    if (c == numZeros[r]) {
                        for (size_t h1 = 0; h1 < H; ++h1) {
                            int oldC = c;
                            if (h1 == 0) {
                                --oldC;
                            }

                            if (oldC >= 0) {
                                alpha[c][h][T - 1] += alpha[oldC][h1][T - 2] * A[h1][h];
                            }
                        }
                        alpha[c][h][T - 1] *= E[h][obs[r][T - 1]];
                    }
                }
            }

            // beta
            std::vector<std::vector<std::vector<double> > >
                beta;  // beta[c][h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta,
                       // c 0's )
            beta.resize(numZeros[r] + 1);
            for (int c = 0; c <= numZeros[r]; ++c) {
                beta[c].resize(H);
                for (size_t h = 0; h < H; ++h) {
                    beta[c][h].resize(T);

                    if (c == 0) {
                        beta[c][h][T - 1] = 1;
                    }

                    else {
                        beta[c][h][T - 1] = 0;
                    }
                }
            }

            for (int t = T - 2; t > 0; --t) {
                for (int c = 0; c <= numZeros[r]; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        beta[c][h][t] = 0.;
                        for (size_t h2 = 0; h2 < H; ++h2) {
                            int newC = c;
                            if (h2 == 0) {
                                --newC;
                            }

                            if (newC >= 0) {
                                beta[c][h][t]
                                    += beta[newC][h2][t + 1] * A[h][h2] * E[h2][obs[r][t + 1]];
                            }
                        }
                    }
                }
            }

            // t = 0
            for (int c = 0; c <= numZeros[r]; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    beta[c][h][0] = 0.;
                }
            }

            // h[0] = 0
            if (numZeros[r] > 0) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    int newC = numZeros[r] - 1;
                    if (h2 == 0) {
                        --newC;
                    }
                    if (newC >= 0) {
                        beta[numZeros[r] - 1][0][0]
                            += beta[newC][h2][1] * A[0][h2] * E[h2][obs[r][1]];
                    }
                }
            }

            // h[0] != 0
            for (size_t h = 1; h < H; ++h) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    int newC = numZeros[r];
                    if (h2 == 0) {
                        --newC;
                    }

                    if (newC >= 0) {
                        beta[numZeros[r]][h][0] += beta[newC][h2][1] * A[h][h2] * E[h2][obs[r][1]];
                    }
                }
            }

            // den = P(O | theta)
            // Need different denominators because of the scaling
            // This is numerically a VERY weird algorithm
            std::vector<double> den;
            for (int t = 0; t < T; ++t) {
                den.push_back(0.);
                for (size_t h = 0; h < H; ++h) {
                    for (int c = 0; c <= numZeros[r]; ++c) {
                        den[t] += alpha[c][h][t] * beta[numZeros[r] - c][h][t];
                    }
                }
            }

            // Gamma
            std::vector<std::vector<double> > gamma;  // gamma[h][t] = P(H_t = h | Y , theta)
            gamma.resize(H);
            for (size_t h = 0; h < H; ++h) {
                gamma[h].resize(T);
            }

            for (size_t h = 0; h < H; ++h) {
                for (int t = 0; t < T; ++t) {
                    double num = 0.;
                    for (int c = 0; c <= numZeros[r]; ++c) {
                        num += alpha[c][h][t] * beta[numZeros[r] - c][h][t];
                    }
                    gamma[h][t] = num / den[t];
                }
            }

            totalGamma.push_back(gamma);

            // xi
            std::vector<std::vector<std::vector<double> > >
                xi;  // xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta)
            xi.resize(H);
            for (size_t h1 = 0; h1 < H; ++h1) {
                xi[h1].resize(H);
                for (size_t h2 = 0; h2 < H; ++h2) {
                    xi[h1][h2].resize(T - 1);
                }
            }

            for (size_t h1 = 0; h1 < H; ++h1) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    for (int t = 0; t < T - 1; ++t) {
                        double num = 0.;

                        for (int c = 0; c <= numZeros[r]; ++c) {
                            int middleC = 0;
                            if (h2 == 0) {
                                ++middleC;
                            }

                            if (numZeros[r] - middleC - c >= 0) {
                                num += alpha[c][h1][t] * beta[numZeros[r] - middleC - c][h2][t + 1];
                            }
                        }
                        num *= A[h1][h2] * E[h2][obs[r][t + 1]];

                        xi[h1][h2][t] = num / den[t];
                    }
                }
            }

            totalXi.push_back(xi);
        }

        // New S
        for (size_t h = 0; h < H; ++h) {
            S[h] = 0.;
            for (size_t r = 0; r < R; ++r) {
                S[h] += totalGamma[r][h][0];
            }
            S[h] /= R;
        }

        // New E
        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t o = 0; o < O; ++o) {
                    double num = 0.;
                    double newDen = 0.;

                    for (int t = 0; t < T; ++t) {
                        if (obs[r][t] == o) {
                            num += totalGamma[r][h][t];
                        }
                        newDen += totalGamma[r][h][t];
                    }

                    E[h][o] = num / newDen;
                }
            }
        }

        double tol = 0.;

        // New A
        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;
                for (size_t r = 0; r < R; ++r) {
                    for (int t = 0; t < T - 1; ++t) {
                        num += totalXi[r][h1][h2][t];
                        newDen += totalGamma[r][h1][t];
                    }
                }
                tol = std::max(std::fabs(A[h1][h2] - num / newDen), tol);
                A[h1][h2] = num / newDen;
            }
        }
        std::cout << "Tolerance: " << tol << "\n";
        // tol = 0.;
        if (tol < eps) {
            break;
        }
    }
}

//---------------------------------------
//-----Learning without Constraints------
//---------------------------------------

void HMM_inference::learn(const std::vector<int> &obs, const double eps)
{
    int T = obs.size();

    while (true) {
        // alpha
        std::vector<std::vector<double> >
            alpha;  // alpha[h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta)
        alpha.resize(H);
        for (size_t h = 0; h < H; ++h) {
            alpha[h].resize(T, 0.);
            alpha[h][0] = S[h] * E[h][obs[0]];
        }

        for (int t = 1; t < T; ++t) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t h1 = 0; h1 < H; ++h1) {
                    alpha[h][t] += alpha[h1][t - 1] * A[h1][h];
                }

                alpha[h][t] *= E[h][obs[t]];
            }
        }

        // beta
        std::vector<std::vector<double> >
            beta;  // beta[h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta)
        beta.resize(H);
        for (size_t h = 0; h < H; ++h) {
            beta[h].resize(T);
            beta[h][T - 1] = 1.;
        }

        for (int t = T - 2; t >= 0; --t) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    beta[h][t] += beta[h2][t + 1] * A[h][h2] * E[h2][obs[t + 1]];
                }
            }
        }

        // den = P(O | theta)
        std::vector<double> den(T, 0);
        for (int t = 0; t < T; ++t) {
            for (size_t h = 0; h < H; ++h) {
                den[t] += alpha[h][t] * beta[h][t];
            }
        }

        // Gamma
        std::vector<std::vector<double> > gamma;  // gamma[h][t] = P(H_t = h | Y , theta)
        gamma.resize(H);
        for (size_t h = 0; h < H; ++h) {
            gamma[h].resize(T);
        }

        for (size_t h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                gamma[h][t] = alpha[h][t] * beta[h][t] / den[t];
            }
        }

        // xi
        std::vector<std::vector<std::vector<double> > >
            xi;  // xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta)
        xi.resize(H);
        for (size_t h1 = 0; h1 < H; ++h1) {
            xi[h1].resize(H);
            for (size_t h2 = 0; h2 < H; ++h2) {
                xi[h1][h2].resize(T - 1);
            }
        }

        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                for (int t = 0; t < T - 1; ++t) {
                    xi[h1][h2][t]
                        = alpha[h1][t] * beta[h2][t + 1] * A[h1][h2] * E[h2][obs[t + 1]] / den[t];
                }
            }
        }

        // New S
        for (size_t h = 0; h < H; ++h) {
            S[h] = gamma[h][0];
        }

        // New E
        for (size_t h = 0; h < H; ++h) {
            for (size_t o = 0; o < O; ++o) {
                double num = 0.;
                double newDen = 0.;

                for (int t = 0; t < T; ++t) {
                    if (obs[t] == o) {
                        num += gamma[h][t];
                    }
                    newDen += gamma[h][t];
                }

                E[h][o] = num / newDen;
            }
        }

        double tol = 0.;

        // New A
        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;

                for (int t = 0; t < T - 1; ++t) {
                    num += xi[h1][h2][t];
                    newDen += gamma[h1][t];
                }
                tol = std::max(std::fabs(A[h1][h2] - num / newDen), tol);
                A[h1][h2] = num / newDen;
            }
        }
        std::cout << "Tolerance: " << tol << "\n";
        // tol = 0.;
        if (tol < eps) {
            break;
        }
    }
}

// Unconstrained learning with multiple observations
// Similar to unconstrained learning with constraints
void HMM_inference::learn(const std::vector<std::vector<int> > &obs, const double eps)
{
    int T = obs[0].size();
    size_t R = obs.size();
    size_t numIt = 0;

    while (true) {
        ++numIt;
        std::vector<std::vector<std::vector<double> > > totalGamma;
        std::vector<std::vector<std::vector<std::vector<double> > > > totalXi;

        for (size_t r = 0; r < R; ++r) {
            // alpha
            std::vector<std::vector<double> >
                alpha;  // alpha[h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta)
            alpha.resize(H);
            for (size_t h = 0; h < H; ++h) {
                alpha[h].resize(T, 0.);
                alpha[h][0] = S[h] * E[h][obs[r][0]];
            }

            for (int t = 1; t < T; ++t) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t h1 = 0; h1 < H; ++h1) {
                        alpha[h][t] += alpha[h1][t - 1] * A[h1][h];
                    }

                    alpha[h][t] *= E[h][obs[r][t]];
                }
            }

            // beta
            std::vector<std::vector<double> >
                beta;  // beta[h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta)
            beta.resize(H);
            for (size_t h = 0; h < H; ++h) {
                beta[h].resize(T);
                beta[h][T - 1] = 1.;
            }

            for (int t = T - 2; t >= 0; --t) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t h2 = 0; h2 < H; ++h2) {
                        beta[h][t] += beta[h2][t + 1] * A[h][h2] * E[h2][obs[r][t + 1]];
                    }
                }
            }

            // den = P(O | theta)
            std::vector<double> den(T, 0);
            for (int t = 0; t < T; ++t) {
                for (size_t h = 0; h < H; ++h) {
                    den[t] += alpha[h][t] * beta[h][t];
                }
            }

            // Gamma
            std::vector<std::vector<double> > gamma;  // gamma[h][t] = P(H_t = h | Y , theta)
            gamma.resize(H);
            for (size_t h = 0; h < H; ++h) {
                gamma[h].resize(T);
            }

            for (size_t h = 0; h < H; ++h) {
                for (int t = 0; t < T; ++t) {
                    gamma[h][t] = alpha[h][t] * beta[h][t] / den[t];
                }
            }
            totalGamma.push_back(gamma);

            // xi
            std::vector<std::vector<std::vector<double> > >
                xi;  // xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta)
            xi.resize(H);
            for (size_t h1 = 0; h1 < H; ++h1) {
                xi[h1].resize(H);
                for (size_t h2 = 0; h2 < H; ++h2) {
                    xi[h1][h2].resize(T - 1);
                }
            }

            for (size_t h1 = 0; h1 < H; ++h1) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    for (int t = 0; t < T - 1; ++t) {
                        xi[h1][h2][t] = alpha[h1][t] * beta[h2][t + 1] * A[h1][h2]
                                        * E[h2][obs[r][t + 1]] / den[t];
                    }
                }
            }
            totalXi.push_back(xi);
        }

        // New S
        for (size_t h = 0; h < H; ++h) {
            S[h] = 0.;
            for (size_t r = 0; r < R; ++r) {
                S[h] += totalGamma[r][h][0];
            }
            S[h] /= R;
        }

        // New E
        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t o = 0; o < O; ++o) {
                    double num = 0.;
                    double newDen = 0.;

                    for (int t = 0; t < T; ++t) {
                        if (obs[r][t] == o) {
                            num += totalGamma[r][h][t];
                        }
                        newDen += totalGamma[r][h][t];
                    }

                    E[h][o] = num / newDen;
                }
            }
        }

        double tol = 0.;

        // New A
        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;
                for (size_t r = 0; r < R; ++r) {
                    for (int t = 0; t < T - 1; ++t) {
                        num += totalXi[r][h1][h2][t];
                        newDen += totalGamma[r][h1][t];
                    }
                }
                tol = std::max(std::fabs(A[h1][h2] - num / newDen), tol);
                A[h1][h2] = num / newDen;
            }
        }

        std::cout << "Tolerance: " << tol << "\n";

        if (tol < eps) {
            break;
        }
    }
}

//------------------------------
//-----Monte Carlo Learning-----
//------------------------------

// Will work best/fastest if the sets of hidden states which satisfy the constraints
// This algorithm is TERRIBLE, I can't even get it to converge in a simple case with T = 10.
// This is currently the only learning algorithm we have for having a constraint oracle rather than
// ``simple'' constraints This also fails to work if we are converging towards values in the
// transition matrix with 0's (which is NOT uncommon)
void HMM_inference::learn(
    const std::vector<std::vector<int> > &obs,
    const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle, const double eps,
    const int C)
{
    size_t R = obs.size();
    if (constraintOracle.size() != R) {
        std::cout << "In learnMC, obs and constraintOracle vectors sizes do not match."
                  << std::endl;
        throw std::exception();
    }
    int totTime = 0.;
    for (size_t r = 0; r < R; ++r) {
        totTime += obs[r].size();
    }

    std::vector<std::vector<double> > SStar;
    std::vector<std::vector<std::vector<double> > > AStar;
    std::vector<std::vector<std::vector<double> > > EStar;

    std::vector<std::vector<int> > SStarCounter;
    std::vector<std::vector<std::vector<int> > > AStarCounter;
    std::vector<std::vector<std::vector<int> > > EStarCounter;

    SStar.resize(R);
    AStar.resize(R);
    EStar.resize(R);
    SStarCounter.resize(R);
    AStarCounter.resize(R);
    EStarCounter.resize(R);

    for (size_t r = 0; r < R; ++r) {
        SStar[r].resize(H);
        AStar[r].resize(H);
        EStar[r].resize(H);
        SStarCounter[r].resize(R);
        AStarCounter[r].resize(R);
        EStarCounter[r].resize(R);

        for (size_t h = 0; h < H; ++h) {
            AStar[r][h].resize(H);
            AStarCounter[r][h].resize(H);
            EStar[r][h].resize(O);
            EStarCounter[r][h].resize(O);
        }
    }

    int totNumIt = 0;
    std::vector<std::vector<std::vector<int> > > allHidden;
    allHidden.resize(R);

    while (true) {
        if ((totNumIt & (totNumIt - 1))
            == 0) {  // Who knows what is best here... this runs if totNumIt is a power of two so
                     // that it becomes more rare as time goes on
            allHidden.clear();
            std::cout << "Generating hidden feasible hidden states randomly.\n";
            int tempCounter = 0;
            for (size_t r = 0; r < R; ++r) {
                size_t numIt = 0;
                std::vector<int> observed;
                std::vector<int> hidden;
                int T = obs[r].size();

                while (numIt <= C * H * std::max(H, O)
                                    / (R * (totTime - 1))) {  // This is so that we have enough
                                                              // counts for A[h][h'] and E[h][o]
                    run(T, observed, hidden);
                    if (constraintOracle[r](hidden)) {
                        allHidden[r].push_back(hidden);
                        ++numIt;
                        ++tempCounter;
                        if ((tempCounter % 1) == 0) {  // This seems like a good pace for printing
                            std::cout << tempCounter << "\n";
                        }
                    }
                }
            }
        }

        ++totNumIt;

        for (size_t r = 0; r < R; ++r) {
            fill(SStar[r].begin(), SStar[r].end(), 0.);
            fill(SStarCounter[r].begin(), SStarCounter[r].end(), 0);
            for (size_t h = 0; h < H; ++h) {
                fill(AStar[r][h].begin(), AStar[r][h].end(), 0.);
                fill(AStarCounter[r][h].begin(), AStarCounter[r][h].end(), 0);
                fill(EStar[r][h].begin(), EStar[r][h].end(), 0.);
                fill(EStarCounter[r][h].begin(), EStarCounter[r][h].end(), 0);
            }
        }

        for (size_t r = 0; r < R; ++r) {
            int T = obs[r].size();
            for (size_t i = 0; i < allHidden[r].size(); ++i) {
                double p = std::exp(logProb(obs[r], allHidden[r][i]));
                // double p = 1.;
                SStar[r][allHidden[r][i][0]] += p;
                ++SStarCounter[r][allHidden[r][i][0]];
                for (int t = 0; t < T - 1; ++t) {
                    AStar[r][allHidden[r][i][t]][allHidden[r][i][t + 1]] += p;
                    ++AStarCounter[r][allHidden[r][i][t]][allHidden[r][i][t + 1]];
                    EStar[r][allHidden[r][i][t]][obs[r][t]] += p;
                    ++EStarCounter[r][allHidden[r][i][t]][obs[r][t]];
                }
                EStar[r][allHidden[r][i][T - 1]][obs[r][T - 1]] += p;
                ++EStarCounter[r][allHidden[r][i][T - 1]][obs[r][T - 1]];
            }
        }

        // Normalize
        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                if (SStarCounter[r][h] == 0) {
                    SStar[r][h] = 0;
                }
                else {
                    SStar[r][h] /= SStarCounter[r][h];
                }
            }
            double SSum = std::accumulate(SStar[r].begin(), SStar[r].end(), 0.);
            for (size_t h = 0; h < H; ++h) {
                SStar[r][h] = SStar[r][h] / SSum;
            }
        }
        for (size_t h = 0; h < H; ++h) {
            double tempSum = 0.;
            for (size_t r = 0; r < R; ++r) {
                tempSum += SStar[r][h];
            }
            S[h] = tempSum / R;
        }

        for (size_t r = 0; r < R; ++r) {
            for (size_t h1 = 0; h1 < H; ++h1) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    if (AStarCounter[r][h1][h2] == 0) {
                        AStar[r][h1][h2] = 0;
                    }
                    else {
                        AStar[r][h1][h2] = AStar[r][h1][h2] / AStarCounter[r][h1][h2];
                    }
                }
            }

            double ASum = 0;
            for (size_t h1 = 0; h1 < H; ++h1) {
                ASum += std::accumulate(AStar[r][h1].begin(), AStar[r][h1].end(), 0.);
            }
            for (size_t h1 = 0; h1 < H; ++h1) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    AStar[r][h1][h2] /= ASum;
                }
            }
        }
        double tol = 0.;
        std::vector<std::vector<double> > newA;
        newA.resize(H);
        for (size_t h1 = 0; h1 < H; ++h1) {
            newA[h1].resize(H);
            for (size_t h2 = 0; h2 < H; ++h2) {
                double tempSum = 0.;
                for (size_t r = 0; r < R; ++r) {
                    tempSum += AStar[r][h1][h2];
                }
                newA[h1][h2] = tempSum;
            }
            double ASum = std::accumulate(newA[h1].begin(), newA[h1].end(), 0.);
            for (size_t h2 = 0; h2 < H; ++h2) {
                if (ASum == 0) {
                    newA[h1][h2] = 1. / H;
                }
                else {
                    newA[h1][h2] = newA[h1][h2] / ASum;
                }
                tol = std::max(std::fabs(A[h1][h2] - newA[h1][h2]), tol);
                A[h1][h2] = newA[h1][h2];
            }
        }

        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t o = 0; o < O; ++o) {
                    if (EStarCounter[r][h][o] == 0) {
                        EStar[r][h][o] = 0;
                    }
                    else {
                        EStar[r][h][o] /= EStarCounter[r][h][o];
                    }
                }
            }

            double ESum = 0.;
            for (size_t h = 0; h < H; ++h) {
                ESum += std::accumulate(EStar[r][h].begin(), EStar[r][h].end(), 0.);
            }
            for (size_t h = 0; h < H; ++h) {
                for (size_t o = 0; o < O; ++o) {
                    EStar[r][h][o] /= ESum;
                }
            }
        }

        for (size_t h = 0; h < H; ++h) {
            for (size_t o = 0; o < O; ++o) {
                double tempSum = 0.;
                for (size_t r = 0; r < R; ++r) {
                    tempSum += EStar[r][h][o];
                }
                E[h][o] = tempSum;
            }
            double ESum = std::accumulate(E[h].begin(), E[h].end(), 0.);
            for (size_t o = 0; o < O; ++o) {
                if (ESum == 0) {
                    E[h][o] = 1. / O;
                }
                else {
                    E[h][o] = E[h][o] / ESum;
                }
            }
        }

        std::cout << "Tolerance: " << tol << "\n";
        // tol = -1;
        if (tol < eps) {
            break;
        }
    }
}

//--------------------
//-----Learn Hard-----
//--------------------

// Uses the hard EM algorithm rather than the soft EM algorithm
// Actually computationally easier than soft EM....
// k is the number of best solutions we wish to use. True hard EM uses k = 1
void HMM_inference::learnHard(
    const std::vector<std::vector<int> > &obs,
    const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle, double eps,
    int numSolns)
{
    size_t R = obs.size();

    std::vector<std::vector<int> > ACounter;
    std::vector<std::vector<int> > ECounter;
    std::vector<int> SCounter;

    ACounter.resize(H);
    ECounter.resize(H);
    SCounter.resize(H);
    for (size_t h = 0; h < H; ++h) {
        ACounter[h].resize(H);
        ECounter[h].resize(O);
    }

    while (true) {
        fill(SCounter.begin(), SCounter.end(), 0);
        for (size_t h = 0; h < H; ++h) {
            fill(ACounter[h].begin(), ACounter[h].end(), 0);
            fill(ECounter[h].begin(), ECounter[h].end(), 0);
        }

        for (size_t r = 0; r < R; ++r) {
            std::cout << "R = " << r << "\n";
            int T = obs[r].size();
            std::vector<std::vector<int> > hidden;
            double temp;

            hidden = aStarMult(obs[r], temp, constraintOracle[r], numSolns);
            for (int i = 0; i < numSolns; ++i) {
                ++SCounter[hidden[i][0]];
                ++ECounter[hidden[i][0]][obs[r][0]];
                for (int t = 1; t < T; ++t) {
                    ++ACounter[hidden[i][t - 1]][hidden[i][t]];
                    ++ECounter[hidden[i][t]][obs[r][t]];
                }
            }
        }

        int sum = 0;
        for (size_t h = 0; h < H; ++h) {
            sum += SCounter[h];
        }
        for (size_t h = 0; h < H; ++h) {
            S[h] = ((double)SCounter[h]) / sum;
        }

        double tol = 0.;
        for (size_t h1 = 0; h1 < H; ++h1) {
            sum = 0;
            for (size_t h2 = 0; h2 < H; ++h2) {
                sum += ACounter[h1][h2];
            }
            for (size_t h2 = 0; h2 < H; ++h2) {
                tol = std::max(std::fabs(A[h1][h2] - ((double)ACounter[h1][h2]) / sum), tol);
                A[h1][h2] = ((double)ACounter[h1][h2]) / sum;
            }
            sum = 0;
            for (size_t o = 0; o < O; ++o) {
                sum += ECounter[h1][o];
            }
            for (size_t o = 0; o < O; ++o) {
                E[h1][o] = ((double)ECounter[h1][o]) / sum;
            }
        }

        std::cout << "Tolerance: " << tol << "\n";
        // tol = -1;
        if (tol < eps) {
            break;
        }
    }
    return;
}

}  // namespace chmmpp

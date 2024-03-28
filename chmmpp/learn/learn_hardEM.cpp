// HMM.cpp

#include <iostream>
#include "learn.hpp"
#include "../inference/inference.hpp"

namespace chmmpp {
void learn_hardEM(HMM &hmm, const std::vector<std::vector<int> > &obs,
                  const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
                  const int numSolns, const double eps)
{
    auto A = hmm.getA();
    auto S = hmm.getS();
    auto E = hmm.getE();
    auto H = hmm.getH();
    auto O = hmm.getO();

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
            std::vector<double> temp;

            aStarMultOracle(hmm, obs[r], hidden, temp, constraintOracle[r], numSolns);
            for (int i = 0; i < hidden.size();
                 ++i) {  // Use hidden.size() here incase there aren't numSolns # of solutions
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
        hmm.setS(S);

        double tol = 0.;
        for (size_t h1 = 0; h1 < H; ++h1) {
            sum = 0;
            for (size_t h2 = 0; h2 < H; ++h2) {
                sum += ACounter[h1][h2];
            }
            for (size_t h2 = 0; h2 < H; ++h2) {
                tol = std::max(std::abs(A[h1][h2] - ((double)ACounter[h1][h2]) / sum), tol);
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
        hmm.setE(E);
        hmm.setA(A);

        std::cout << "Tolerance: " << tol << "\n";

        if (tol < eps) {
            break;
        }
    }
    return;
}

void learn_hardEM(HMM &hmm, const std::vector<int> &obs,
                  const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns,
                  const double eps)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    std::vector<std::function<bool(std::vector<int>)> > newConstraintOracle;
    newConstraintOracle.push_back(constraintOracle);
    learn_stochastic(hmm, newObs, newConstraintOracle, numSolns, eps);
}

}  // namespace chmmpp

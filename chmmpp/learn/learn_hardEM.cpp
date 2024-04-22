#include <iostream>
#include "learn.hpp"
#include "../inference/inference.hpp"

namespace chmmpp {

namespace {

void process_options(const Options& options, double& convergence_tolerance,
                     unsigned int& max_iterations)
{
    for (const auto& it : options.options) {
        if (it.first == "max_iterations") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp > 0)
                    max_iterations = tmp;
                else
                    std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer"
                              << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second)) {
                max_iterations = std::get<unsigned int>(it.second);
            }
            else
                std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer"
                          << std::endl;
        }
        else if (it.first == "convergence_tolerance") {
            if (std::holds_alternative<double>(it.second))
                convergence_tolerance = std::get<double>(it.second);
            else
                std::cerr << "WARNING: 'convergence_tolerance' option must be a double"
                          << std::endl;
        }
    }
}

}  // namespace

void learn_hardEM(HMM& hmm, const std::vector<std::vector<int> >& obs,
                  const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle,
                  const int numSolns, const Options& options)
{
    double convergence_tolerance = 10E-6;
    unsigned int max_iterations = 0;
    process_options(options, convergence_tolerance, max_iterations);

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

    size_t iter = 0;
    while (true) {
        fill(SCounter.begin(), SCounter.end(), 0);
        for (size_t h = 0; h < H; ++h) {
            fill(ACounter[h].begin(), ACounter[h].end(), 0);
            fill(ECounter[h].begin(), ECounter[h].end(), 0);
        }

        for (size_t r = 0; r < R; ++r) {
            // std::cout << "R = " << r << "\n";
            int T = obs[r].size();
            std::vector<std::vector<int> > hidden;
            std::vector<double> temp;

            aStarMultOracle(hmm, obs[r], hidden, temp, constraintOracle[r], numSolns,
                            max_iterations);
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

        if (tol < convergence_tolerance) break;
        if (++iter >= max_iterations) break;
    }
}

void learn_hardEM(HMM& hmm, const std::vector<int>& obs,
                  const std::function<bool(std::vector<int>&)>& constraintOracle,
                  const int numSolns, const Options& options)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    std::vector<std::function<bool(std::vector<int>&)> > newConstraintOracle;
    newConstraintOracle.push_back(constraintOracle);
    // WEH - Is this an error???
    // learn_stochastic(hmm, newObs, newConstraintOracle, numSolns, options);
    learn_hardEM(hmm, newObs, newConstraintOracle, numSolns, options);
}

}  // namespace chmmpp

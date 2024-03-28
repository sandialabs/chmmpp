// main.cpp

#include <iostream>
#include "numZerosHMM.hpp"

template <typename T, typename V, typename W, typename Z>
void run(T& hmm, V& obs, W& hid, size_t numSolns, const Z& fn)
{
    double logProb;
    std::vector<std::vector<int>> hidGuess;
    fn(hmm, obs, hidGuess, logProb, numSolns);


    std::cout << "Top " << numSolns << " solutions.\n";
    for (size_t r = 0; r < numSolns; ++r) {
        std::cout << "  Solution:";
        for (auto& v : hidGuess[r])
            std::cout << v;
        std::cout << "\n";

        int numDiff= 0;
        for (size_t t = 0; t < obs.size(); ++t) {
            if (hidGuess[r][t] != hid[t]) {
                ++numDiff;
            }
        }
        std::cout << "  Num zeros:                       " << count(hidGuess[r].begin(), hidGuess[r].end(), 0) << "\n";
        std::cout << "  Number of mistakes in inference: " << numDiff << "\n";
        std::cout << "  Log prob:                        " << hmm.logProb(obs, hidGuess[r]) << std::endl;
        std::cout << std::endl;
    }

    std::cout << "OPTIMAL Log prob: " << -logProb << "\n";
    std::cout << std::endl;
}

int main()
{
    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.501, 0.499};                               // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 25;        // Time Horizon
    size_t numSolns = 5;  // Find top # of solutions

    chmmpp::HMM hmm(A, S, E, 1937309487);
    hmm.print();

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<int> obs;
    std::vector<int> hid;

    hmm.run(T, obs, hid);
    auto numZeros = count(hid.begin(), hid.end(), 0);
    std::cout << "Num Zeros in randomly generated data: " << numZeros << std::endl << std::endl;

    std::cout << "Observed:\n";
    for (auto& v : obs)
        std::cout << v;
    std::cout << std::endl;

    std::cout << "\nTrue solution:\n";
    for (auto& v : hid)
        std::cout << v;
    std::cout << std::endl << std::endl;;

    chmmpp::numZerosHMM nzhmm(numZeros);
    nzhmm.initialize(hmm);


    //std::cout << "Running inference without constraint - aStarMulti\n";
    //run(hmm, obs, hid, numSolns, [](chmmpp::HMM& hmm, const std::vector<int>& obs, std::vector<int>& hs, double& logProb, size_t num){hmm.aStarMult(obs,hs,logProb,num);});

    std::cout << "Running inference with constraint - custom aStar\n";
    run(nzhmm, obs, hid, numSolns, [](chmmpp::numZerosHMM& hmm, const std::vector<int>& obs, std::vector<std::vector<int>>& hs, double& logProb, size_t num){hmm.aStarMult_numZeros(obs,hs,logProb,num);});

    return 0;
}

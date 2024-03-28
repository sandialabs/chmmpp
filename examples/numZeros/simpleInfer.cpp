// main.cpp

#include <iostream>
#include "numZerosHMM.hpp"

template <typename T, typename V, typename W, typename Z>
void run(T& hmm, V& obs, W& hid, const Z& fn)
{
    double logProb;
    std::vector<int> hidGuess;
    fn(hmm, obs, hidGuess, logProb);

    int numDiff = 0;
    for (size_t t = 0; t < obs.size(); ++t) {
        if (hidGuess[t] != hid[t]) {
            ++numDiff;
        }
    }

    std::cout << "  Solution: ";
    for (auto& v : hidGuess) std::cout << v;
    std::cout << "\n";

    std::cout << "  Log prob:                        " << -logProb << "\n";
    std::cout << "  Double-checking log prob:        " << hmm.logProb(obs, hidGuess) << std::endl;
    std::cout << "  Num zeros:                       " << count(hidGuess.begin(), hidGuess.end(), 0)
              << "\n";
    std::cout << "  Number of mistakes in inference: " << numDiff << "\n";
    std::cout << std::endl;
}

int main()
{
    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.9, 0.1};                                   // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 25;  // Time Horizon

    chmmpp::HMM hmm(A, S, E, 1937309487);
    hmm.print();

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<int> obs;
    std::vector<int> hid;

    hmm.run(T, obs, hid);
    auto numZeros = count(hid.begin(), hid.end(), 0);
    std::cout << "Num Zeros in randomly generated data: " << numZeros << std::endl << std::endl;

    chmmpp::numZerosHMM nzhmm(numZeros);
    nzhmm.initialize(hmm);

    std::cout << "Observed:\n";
    for (auto& v : obs) std::cout << v;
    std::cout << std::endl;

    std::cout << "\nTrue solution:\n";
    for (auto& v : hid) std::cout << v;
    std::cout << std::endl << std::endl;
    ;

    std::cout << "Running inference without constraint - aStar\n";
    run(hmm, obs, hid,
        [](chmmpp::HMM& hmm, const std::vector<int>& obs, std::vector<int>& hs, double& logProb) {
            hmm.aStar(obs, hs, logProb);
        });
    std::cout << "Running inference without constraint - Viterbi\n";
    run(hmm, obs, hid,
        [](chmmpp::HMM& hmm, const std::vector<int>& obs, std::vector<int>& hs, double& logProb) {
            hmm.viterbi(obs, hs, logProb);
        });

    std::cout << "Running inference without constraint - LP\n";
    run(hmm, obs, hid,
        [](chmmpp::HMM& hmm, const std::vector<int>& obs, std::vector<int>& hs, double& logProb) {
            hmm.lp_map_inference(obs, hs, logProb);
        });

    std::cout << "Running inference with constraint - custom aStar\n";
    run(nzhmm, obs, hid,
        [](chmmpp::numZerosHMM& hmm, const std::vector<int>& obs, std::vector<int>& hs,
           double& logProb) { hmm.aStar_numZeros(obs, hs, logProb); });

    std::cout << "Running inference with constraint - generic aStar\n";
    run(nzhmm, obs, hid,
        [](chmmpp::numZerosHMM& hmm, const std::vector<int>& obs, std::vector<int>& hs,
           double& logProb) { hmm.aStar(obs, hs, logProb); });

    std::cout << "Running inference with constraint - MIP \n";
    run(nzhmm, obs, hid,
        [](chmmpp::numZerosHMM& hmm, const std::vector<int>& obs, std::vector<int>& hs,
           double& logProb) { hmm.mip_map_inference(obs, hs, logProb); });

    return 0;
}

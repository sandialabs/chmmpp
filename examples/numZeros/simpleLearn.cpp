//
// Learn HMM parameters
//
// Generating random trials where the number of nonzeros is fixed
//
#include <iostream>
#include "numZerosHMM.hpp"

template <typename T, typename V, typename Z>
void run(T& hmm, V& obs, const Z& fn)
{
    double logProb;
    std::vector<int> hidGuess;
    fn(hmm, obs);

    hmm.print();
}

void run_all(bool with_rejection)
{
    // Initial Guess
    std::vector<std::vector<double>> A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.9, 0.1};                                  // Start probabilities
    std::vector<std::vector<double>> E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 25;         // Time Horizon
    size_t numIt = 10000;     // Number of runs
    size_t numZeros = 10;  // Number of zeros in the hidden states

    chmmpp::HMM hmm(A, S, E, 1937309487);

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<std::vector<int>> obs(numIt);
    std::vector<std::vector<int>> hid(numIt);

    std::cout << "Num Obs:   " << T << std::endl;
    std::cout << "Num Runs:  " << numIt << std::endl;
    std::cout << "Num Zeros: " << numZeros << std::endl << std::endl;
    hmm.reset_rng();
    for (size_t i = 0; i < numIt; ++i) {
        bool feasible = false;
        while (not feasible) {
            hmm.run(T, obs[i], hid[i]);
            feasible = count(hid[i].begin(), hid[i].end(), 0) == numZeros;
            if (not with_rejection) break;
        }

#if 0
        std::cout << "Trial: " << i << std::endl;
        std::cout << "Observed:      ";
        for (auto& v : obs[i]) std::cout << v;
        std::cout << std::endl;

        std::cout << "Hidden states: ";
        for (auto& v : hid[i]) std::cout << v;
        std::cout << std::endl;
        std::cout << "Num zeros: " << count(hid[i].begin(), hid[i].end(), 0) << std::endl;
        std::cout << std::endl;
#endif
    }

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Initial HMM parameters\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmm.print();

    chmmpp::numZerosHMM nzhmm(numZeros);
    nzhmm.initialize(hmm);

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning without constraint - ML estimate using hidden states\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmm.estimate_hmm(obs, hid);
    hmm.print();

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning without constraint - Baum-Welch\n";
    std::cout << "------------------------------------------------------------------------\n";
    run(hmm, obs,
        [](chmmpp::HMM& hmm, const std::vector<std::vector<int>>& obs) { hmm.baum_welch(obs); });

    std::cout << std::endl;
}

int main()
{
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " Generating samples without rejection\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    run_all(false);

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " Generating samples with rejection\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    run_all(true);

    return 0;
}

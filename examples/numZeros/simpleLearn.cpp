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

int main()
{
    // Initial Guess
    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.9, 0.1};                                   // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 25;          // Time Horizon
    size_t numIt = 10;      // Number of runs
    size_t numZeros = 10;   // Number of zeros in the hidden states

    chmmpp::HMM hmm(A, S, E, 1937309487);
    std::cout << "Initial Guess" << std::endl;
    hmm.print();

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<std::vector<int>> obs(numIt);
    std::vector<std::vector<int>> hid(numIt);

    std::cout << "Num Runs:  " << numIt << std::endl << std::endl;
    std::cout << "Num Zeros: " << numZeros << std::endl << std::endl;
    for (size_t i=0; i<numIt; ++i) {

        bool feasible=false;
        while (not feasible) {
            hmm.run(T, obs[i], hid[i]);
            feasible = count(hid[i].begin(), hid[i].end(), 0) == numZeros;
        }

    std::cout << "Trial: " << i << std::endl;
    std::cout << "Observed:      ";
    for (auto& v : obs[i]) std::cout << v;
    std::cout << std::endl;

    std::cout << "Hidden states: ";
    for (auto& v : hid[i]) std::cout << v;
    std::cout << std::endl;
    std::cout << "Num zeros: " << count(hid[i].begin(), hid[i].end(), 0) << std::endl;
    std::cout << std::endl;
    }

    chmmpp::numZerosHMM nzhmm(numZeros);
    nzhmm.initialize(hmm);


    std::cout << "Running learning without constraint - Baum-Welch\n";
    run(hmm, obs, 
        [](chmmpp::HMM& hmm, const std::vector<std::vector<int>>& obs) {
            hmm.baum_welch(obs);
        });

    return 0;
}

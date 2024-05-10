//
// Learn HMM parameters
//
// Generating random trials where the number of nonzeros is fixed
//
#include <iostream>
#include <coek/coek.hpp>
#include "numZerosHMM.hpp"

template <typename T, typename V, typename Z>
void run(T& hmm, V& obs, const Z& fn)
{
    coek::tic();
    hmm.reset_rng();
    fn(hmm, obs);
    auto tdiff = coek::toc();

    std::cout << "Time (sec): " << tdiff << std::endl << std::endl;
    hmm.print_options();
    std::cout << std::endl;

    std::cout << "HMM Parameters" << std::endl;
    std::cout << "~~~~~~~~~~~~~~" << std::endl;
    hmm.print();
}

void run_all(bool with_rejection, bool debug = false)
{
    size_t T = 25;         // Time Horizon
    //size_t numIt = 5000;   // Number of runs
    size_t numIt = 50;   // Number of runs
    size_t numZeros = 10;  // Number of zeros in the hidden states
    size_t seed = 1937309487;
    std::cout << "Num Obs:   " << T << std::endl;
    std::cout << "Num Runs:  " << numIt << std::endl;
    std::cout << "Num Zeros: " << numZeros << std::endl << std::endl;

    // Initial guess of HMM parameters
    std::vector<std::vector<double>> A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.9, 0.1};                                  // Start probabilities
    std::vector<std::vector<double>> E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    // Create HMM
    chmmpp::HMM hmm(A, S, E, seed);
    chmmpp::HMM original_hmm = hmm;

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<std::vector<int>> obs(numIt);
    std::vector<std::vector<int>> hid(numIt);

    hmm.reset_rng();
    for (size_t i = 0; i < numIt; ++i) {
        bool feasible = false;

        if (with_rejection) {
            // Iterate until we generate a sample with numZeros zeros
            while (not feasible) {
                hmm.run(T, obs[i], hid[i]);
                feasible = count(hid[i].begin(), hid[i].end(), 0) == numZeros;
            }
        }
        else {
            // Generate sequence of hidden states and observables
            // Don't worry if the observations match numZeros
            hmm.run(T, obs[i], hid[i]);
            feasible = count(hid[i].begin(), hid[i].end(), 0) == numZeros;
        }

        if (debug) {
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
    }

    chmmpp::numZerosHMM nzhmm(numZeros);

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Initial HMM parameters\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmm.print();

    // HMM Tests

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning without constraint - ML estimate using hidden states\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmm = original_hmm;
    hmm.estimate_hmm(obs, hid);
    hmm.print();

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning without constraint - Baum-Welch\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmm = original_hmm;
    run(hmm, obs,
        [](chmmpp::HMM& hmm, const std::vector<std::vector<int>>& obs) { hmm.baum_welch(obs); });

    // CHMM Tests

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - Soft EM\n";
    std::cout << "------------------------------------------------------------------------\n";
    nzhmm.initialize(original_hmm);
    run(nzhmm, obs, [](chmmpp::CHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_stochastic(obs);
    });

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - Hard EM\n";
    std::cout << "------------------------------------------------------------------------\n";
    nzhmm.initialize(original_hmm);
    nzhmm.set_option("max_iterations", 100);
    run(nzhmm, obs, [](chmmpp::CHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_hardEM(obs);
    });
    nzhmm.clear_options();

    // NZHMM Tests

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - Customized Soft EM???\n";
    std::cout << "------------------------------------------------------------------------\n";
    nzhmm.initialize(original_hmm);
    run(nzhmm, obs, [](chmmpp::numZerosHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_numZeros(obs);
    });

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - SAEM with MIP\n";
    std::cout << "------------------------------------------------------------------------\n";
    nzhmm.initialize(original_hmm);
    //nzhmm.set_option("debug", 1);
    run(nzhmm, obs, [](chmmpp::numZerosHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_mip(obs);
    //nzhmm.reset_options();
    });

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

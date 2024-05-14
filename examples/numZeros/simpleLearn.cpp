//
// Learn HMM parameters
//
// Generating random trials where the number of nonzeros is fixed
//
#include <iostream>
#include <coek/coek.hpp>
#include "numZerosHMM.hpp"

template <typename T, typename V, typename Z>
void run(T& hmm, V& observations, const Z& fn)
{
    std::cout << "Initial HMM Parameters" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    hmm.print();

    hmm.print_options();
    std::cout << std::endl;

    coek::tic();
    hmm.reset_rng();
    fn(hmm, observations);
    auto tdiff = coek::toc();

    std::cout << "Time (sec): " << tdiff << std::endl << std::endl;

    std::cout << "Final HMM Parameters" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~" << std::endl;
    hmm.print();

    hmm.clear_options();
}

void run_all(bool with_rejection, bool debug = false)
{
#if 1
    size_t T = 25;         // Time Horizon
    size_t numObs = 50;   // Number of runs
    size_t numZeros = 10;  // Number of zeros in the hidden states
#else
    size_t T = 5;         // Time Horizon
    size_t numObs = 5;   // Number of runs
    size_t numZeros = 2;  // Number of zeros in the hidden states
#endif
    unsigned int seed = 1937309487;
    std::cout << "Time Horizon:     " << T << std::endl;
    std::cout << "Num Observations: " << numObs << std::endl;
    std::cout << "Num Zeros:        " << numZeros << std::endl << std::endl;

    // Initial guess of HMM parameters
    std::vector<std::vector<double>> A{{0.6, 0.4}, {0.3, 0.7}};  // Transition Matrix
    std::vector<double> S = {0.4, 0.6};                          // Start probabilities
    //std::vector<std::vector<double>> E{{0.6, 0.4}, {0.2, 0.8}};  // Emission Matrix
    //std::vector<std::vector<double>> A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    //std::vector<double> S = {0.9, 0.1};                                  // Start probabilities
    std::vector<std::vector<double>> E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    // Create HMM
    chmmpp::HMM hmm(A, S, E, seed);
    chmmpp::HMM original_hmm = hmm;

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<std::vector<int>> obs(numObs);
    std::vector<std::vector<int>> hid(numObs);

    hmm.reset_rng();
    size_t nsamples=0;
    for (size_t i = 0; i < numObs; ++i) {
        bool feasible = false;

        if (with_rejection) {
            // Iterate until we generate a sample with numZeros zeros
            while (not feasible) {
                hmm.run(T, obs[i], hid[i]);
                feasible = count(hid[i].begin(), hid[i].end(), 0) == numZeros;
                nsamples++;
            }
        }
        else {
            // Generate sequence of hidden states and observables
            // Don't worry if the observations match numZeros
            hmm.run(T, obs[i], hid[i]);
            feasible = count(hid[i].begin(), hid[i].end(), 0) == numZeros;
            nsamples++;
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

    std::cout << "Num Samples of Hidden States: " << nsamples << std::endl;
    std::cout << "Acceptance Rate: " << ((double)numObs)/nsamples << std::endl << std::endl;

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

#if 0
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
    nzhmm.set_option("max_iterations", 1000);
    run(nzhmm, obs, [](chmmpp::CHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_stochastic(obs);
    });

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - Hard EM\n";
    std::cout << "------------------------------------------------------------------------\n";
    nzhmm.initialize(original_hmm);
    nzhmm.set_option("max_iterations", 1000);
    run(nzhmm, obs, [](chmmpp::CHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_hardEM(obs);
    });

    // NZHMM Tests

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - Customized Soft EM???\n";
    std::cout << "------------------------------------------------------------------------\n";
    nzhmm.initialize(original_hmm);
    nzhmm.set_option("max_iterations", 1000);
    run(nzhmm, obs, [](chmmpp::numZerosHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_numZeros(obs);
    });

#endif

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - SAEM with MIP\n";
    std::cout << "------------------------------------------------------------------------\n";
    nzhmm.initialize(original_hmm);
    nzhmm.set_option("max_iterations", 100);
    nzhmm.set_option("select", 1);
    nzhmm.set_option("quiet", 0);
    nzhmm.set_option("debug", 0);
    run(nzhmm, obs, [](chmmpp::numZerosHMM& nzhmm, const std::vector<std::vector<int>>& obs) {
        nzhmm.learn_mip(obs);
    });

    std::cout << std::endl;
}

int main()
{
    bool debug=true;


#if 0
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " Generating samples without rejection\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    run_all(false, debug);
#endif

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " Generating samples with rejection\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    run_all(true, debug);

    return 0;
}

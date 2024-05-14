//
// Infer HMM hidden states
//
#include <iostream>
#ifdef WITH_COEK
#include <coek/coek.hpp>
#endif
#include "numZerosHMM.hpp"

template <typename T, typename V, typename W, typename Z>
void run(T& hmm, V& obs, W& hid, const Z& fn)
{
#ifdef WITH_COEK
    coek::tic();
#endif
    double logProb;
    std::vector<int> hidGuess;
    fn(hmm, obs, hidGuess, logProb);

    int numDiff = 0;
    for (size_t t = 0; t < obs.size(); ++t) {
        if (hidGuess[t] != hid[t]) {
            ++numDiff;
        }
    }
#ifdef WITH_COEK
    auto tdiff = coek::toc();
#else
    double tdiff = 0.0;
#endif

    hmm.print_options();
    std::cout << std::endl;

    std::cout << "  Solution: ";
    for (auto& v : hidGuess) std::cout << v;
    std::cout << "\n";

    std::cout << "  Log prob:                        " << -logProb << "\n";
    std::cout << "  Double-checking log prob:        " << hmm.logProb(obs, hidGuess) << std::endl;
    std::cout << "  Num zeros:                       " << count(hidGuess.begin(), hidGuess.end(), 0)
              << "\n";
    std::cout << "  Number of mistakes in inference: " << numDiff << "\n";
    std::cout << "  Time (sec):                      " << tdiff << "\n";
    std::cout << std::endl;
}

int main()
{
    // Top-level config parameters
    size_t T = 25;                  // Time Horizon
    size_t inference_numZeros = 0;  // Number of zeros used for inference.
    size_t seed = 1937309487;

    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.9, 0.1};                                   // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    // Create HMM
    chmmpp::HMM hmm(A, S, E, seed);
    hmm.print();

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<int> obs;
    std::vector<int> hid;

    // Generate sequence of hidden states and observables
    hmm.run(T, obs, hid);
    auto numZeros = count(hid.begin(), hid.end(), 0);
    if (inference_numZeros == 0) inference_numZeros = numZeros;

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Observed:\n";
    for (auto& v : obs) std::cout << v;
    std::cout << std::endl;

    std::cout << "\nTrue solution:\n";
    for (auto& v : hid) std::cout << v;
    std::cout << std::endl << std::endl;
    ;

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Num Zeros in randomly generated data: " << numZeros << std::endl;
    std::cout << "Target num Zeros for inference:       " << numZeros << std::endl;
    std::cout << "------------------------------------------------------------------------\n\n";

    // Configure the numZerosHMM object for inference
    chmmpp::numZerosHMM nzhmm(inference_numZeros);
    nzhmm.initialize(hmm);

    // HMM Tests

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running inference without constraint - aStar\n";
    std::cout << "------------------------------------------------------------------------\n";
    run(hmm, obs, hid,
        [](chmmpp::HMM& hmm, const std::vector<int>& obs, std::vector<int>& hs, double& logProb) {
            hmm.aStar(obs, hs, logProb);
        });
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running inference without constraint - Viterbi\n";
    std::cout << "------------------------------------------------------------------------\n";
    run(hmm, obs, hid,
        [](chmmpp::HMM& hmm, const std::vector<int>& obs, std::vector<int>& hs, double& logProb) {
            hmm.viterbi(obs, hs, logProb);
        });

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running inference without constraint - LP\n";
    std::cout << "------------------------------------------------------------------------\n";
    run(hmm, obs, hid,
        [](chmmpp::HMM& hmm, const std::vector<int>& obs, std::vector<int>& hs, double& logProb) {
            hmm.lp_map_inference(obs, hs, logProb);
        });

    // NZHMM Tests

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running inference with constraint - custom aStar\n";
    std::cout << "------------------------------------------------------------------------\n";
    run(nzhmm, obs, hid,
        [](chmmpp::numZerosHMM& nzhmm, const std::vector<int>& obs, std::vector<int>& hs,
           double& logProb) { nzhmm.aStar_numZeros(obs, hs, logProb); });

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running inference with constraint - generic aStar\n";
    std::cout << "------------------------------------------------------------------------\n";
    run(nzhmm, obs, hid,
        [](chmmpp::numZerosHMM& nzhmm, const std::vector<int>& obs, std::vector<int>& hs,
           double& logProb) { nzhmm.aStar(obs, hs, logProb); });

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running inference with constraint - MIP \n";
    std::cout << "------------------------------------------------------------------------\n";
    // nzhmm.set_option("debug", 1);
    run(nzhmm, obs, hid,
        [](chmmpp::numZerosHMM& nzhmm, const std::vector<int>& obs, std::vector<int>& hs,
           double& logProb) { nzhmm.mip_map_inference(obs, hs, logProb); });

    return 0;
}

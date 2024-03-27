// main.cpp

#include <iostream>
#include <chmmpp/chmmpp.hpp>

int main()
{
    // bool oracleConstraint(std::vector<int> hid, double numZeros);

    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.9, 0.1};                                   // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 1000;  // Time Horizon

    chmmpp::HMM myHMM(A, S, E, 0);
    myHMM.print();

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<int> obs;
    std::vector<int> hid;

    myHMM.run(T, obs, hid);
    auto numZeros = count(hid.begin(), hid.end(), 0);

    std::cout << "Running inference without constraint - aStar\n";
    double logProbNoConstraints_aStar;
    std::vector<int> hidGuessNoConstraints_aStar;
    aStar(myHMM, obs, hidGuessNoConstraints_aStar, logProbNoConstraints_aStar);

    std::cout << "Running inference without constraint - Viterbi\n";
    double logProbNoConstraints_viterbi;
    std::vector<int> hidGuessNoConstraints_viterbi;
    viterbi(myHMM, obs, hidGuessNoConstraints_viterbi, logProbNoConstraints_viterbi);

    std::cout << "Running inference without constraint - lp\n";
    double logProbNoConstraints_lp;
    std::vector<int> hidGuessNoConstraints_lp;
    lp_map_inference(myHMM, obs, hidGuessNoConstraints_lp, logProbNoConstraints_lp);

    std::cout << "Running inference with constraints.\n";
    double logProbConstraints;
    std::vector<int> hidGuessConstraints;
    aStar_numZeros(myHMM, obs, hidGuessConstraints, logProbConstraints, numZeros);

    // std::vector<int> hidGuessConstraints = myHMM.aStarOracle(obs, logProbConstraints,
    // [numZeros](std::vector<int> myHid) -> bool { return (numZeros == count(myHid.begin(),
    // myHid.end(), 0));  }); //Gives the same answer as above, but slower. This inference method
    // works better if we don't have a ``nice'' constraint like numZeros. It uses an oracles and
    // just tells you at the end if you satisfy the constraints.

    int numDiffNoConstraints_aStar = 0;
    int numDiffNoConstraints_viterbi = 0;
    int numDiffNoConstraints_lp = 0;
    int numDiffConstraints = 0;
    for (size_t t = 0; t < T; ++t) {
        if (hidGuessNoConstraints_aStar[t] != hid[t]) {
            ++numDiffNoConstraints_aStar;
        }
        if (hidGuessNoConstraints_viterbi[t] != hid[t]) {
            ++numDiffNoConstraints_viterbi;
        }
        if (hidGuessNoConstraints_lp[t] != hid[t]) {
            ++numDiffNoConstraints_lp;
        }
        if (hidGuessConstraints[t] != hid[t]) {
            ++numDiffConstraints;
        }
    }

    std::cout << std::endl;
    std::cout << "Log prob without constraints - aStar:\t" << -logProbNoConstraints_aStar << "\n";
    std::cout << "Log prob without constraints - viterbi:\t" << -logProbNoConstraints_viterbi << "\n";
    std::cout << "Log prob without constraints - lp:\t" << -logProbNoConstraints_lp << "\n";
    std::cout << "Log prob with constraints:\t\t" << logProbConstraints << "\n";

    std::cout << std::endl;
    std::cout << "Number of mistakes in inference with no constraints - aStar:\t"
              << numDiffNoConstraints_aStar << "\n";
    std::cout << "Number of mistakes in inference with no constraints - viterbi:\t"
              << numDiffNoConstraints_viterbi << "\n";
    std::cout << "Number of mistakes in inference with no constraints - lp:\t"
              << numDiffNoConstraints_lp << "\n";
    std::cout << "Number of mistakes in inference with constraints:\t\t" << numDiffConstraints
              << "\n";

    std::cout << std::endl;
    std::cout << "Double-checking log prob without constraints - aStar:\t"
              << myHMM.logProb(obs, hidGuessNoConstraints_aStar) << std::endl;
    std::cout << "Double-checking log prob without constraints - viterbi:\t"
              << myHMM.logProb(obs, hidGuessNoConstraints_viterbi) << std::endl;
    std::cout << "Double-checking log prob without constraints - lp:\t"
              << myHMM.logProb(obs, hidGuessNoConstraints_lp) << std::endl;
    std::cout << "Double-checking log prob with constraints:\t\t"
              << myHMM.logProb(obs, hidGuessConstraints) << std::endl;

    return 0;
}

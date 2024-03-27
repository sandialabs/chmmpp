// main.cpp

#include <iostream>
#include <chmmpp/chmmpp.hpp>

bool oracleConstraint(std::vector<int> hid, long int numZeros)
{
    return (numZeros == count(hid.begin(), hid.end(), 0));
}

int main()
{
    // bool oracleConstraint(std::vector<int> hid, double numZeros);

    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.501, 0.499};                               // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 25;        // Time Horizon
    size_t numSolns = 5;  // Find top # of solutions

    chmmpp::HMM myHMM(A, S, E, 1234);  // 1234 is the seed

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<int> obs;
    std::vector<int> hid;

    myHMM.run(T, obs, hid);
    auto numZeros = count(hid.begin(), hid.end(), 0);

    std::cout << "Running inference without constraint.\n";
    double logProbNoConstraints;
    std::vector<std::vector<int> > hidGuessNoConstraints;
    chmmpp::aStarMultOracle(
        myHMM, obs, hidGuessNoConstraints, logProbNoConstraints,
        [](std::vector<int> /*myHid*/) -> bool { return true; }, numSolns);

    std::cout << "Running inference with constraints.\n";
    double logProbConstraints;
    std::vector<std::vector<int> > hidGuessConstraints;
    chmmpp::aStarMultOracle(
        myHMM, obs, hidGuessConstraints, logProbConstraints,
        [numZeros](std::vector<int> myHid) -> bool {
            return (numZeros == count(myHid.begin(), myHid.end(), 0));
        },
        numSolns);

    std::cout << "Observed:\n";
    for (size_t t = 0; t < T; ++t) {
        std::cout << obs[t];
    }

    std::cout << "\n\nTrue solution:\n";
    for (size_t t = 0; t < T; ++t) {
        std::cout << hid[t];
    }
    std::cout << "\n\nTop " << numSolns << " solutions with no constraints.\n";
    for (size_t r = 0; r < numSolns; ++r) {
        for (size_t t = 0; t < T; ++t) {
            std::cout << hidGuessNoConstraints[r][t];
        }
        std::cout << "\n";
    }

    std::cout << "\n\nTop " << numSolns << " solutions with constraints.\n";
    for (size_t r = 0; r < numSolns; ++r) {
        for (size_t t = 0; t < T; ++t) {
            std::cout << hidGuessConstraints[r][t];
        }
        std::cout << "\n";
    }

    return 0;
}

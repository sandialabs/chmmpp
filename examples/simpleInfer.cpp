// main.cpp

#include <iostream>
#include <chmmpp/chmmpp.hpp>

int main()
{
    // bool oracleConstraint(std::vector<int> hid, double numZeros);

    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.9, 0.1};                               // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 10;  // Time Horizon

    chmmpp::HMM myHMM(A, S, E, 0);
    myHMM.print();

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<int> obs;
    std::vector<int> hid;

    myHMM.run(T, obs, hid);
    auto numZeros = count(hid.begin(), hid.end(), 0);

    std::cout << "Running inference without constraint.\n";
    double logProbNoConstraints;
    std::vector<int> hidGuessNoConstraints;
    aStar(myHMM, obs, hidGuessNoConstraints, logProbNoConstraints);

    std::cout << "Running inference with constraints.\n";
    double logProbConstraints;
    std::vector<int> hidGuessConstraints;
    aStar_numZeros(myHMM, obs, hidGuessConstraints, logProbConstraints, numZeros);

    // std::vector<int> hidGuessConstraints = myHMM.aStarOracle(obs, logProbConstraints,
    // [numZeros](std::vector<int> myHid) -> bool { return (numZeros == count(myHid.begin(),
    // myHid.end(), 0));  }); //Gives the same answer as above, but slower. This inference method
    // works better if we don't have a ``nice'' constraint like numZeros. It uses an oracles and
    // just tells you at the end if you satisfy the constraints.

    int numDiffNoConstraints = 0;
    int numDiffConstraints = 0;
    for (size_t t = 0; t < T; ++t) {
        if (hidGuessNoConstraints[t] != hid[t]) {
            ++numDiffNoConstraints;
        }
        if (hidGuessConstraints[t] != hid[t]) {
            ++numDiffConstraints;
        }
    }

    std::cout << "\nLog prob without constraints: " << -logProbNoConstraints << "\n";
    std::cout << "Log prob with constraints: " << logProbConstraints << "\n\n";
    std::cout << "Number of mistakes in inference with no constraints: " << numDiffNoConstraints
              << "\n";
    std::cout << "Number of mistakes in inference with constraints: " << numDiffConstraints
              << "\n\n";

    std::cout << "Double-checking log prob without constraints: " << myHMM.logProb(obs, hidGuessNoConstraints) << std::endl;
    std::cout << "Double-checking log prob with constraints:    " << myHMM.logProb(obs, hidGuessConstraints) << std::endl;

    return 0;
}

// main.cpp

#include <iostream>
#include <chmmpp/chmmpp.hpp>

int main()
{
    // bool oracleConstraint(std::vector<int> hid, double numZeros);

    std::vector<std::vector<double> > A{{0.899, 0.101}, {0.099, 0.901}};  // Transition Matrix
    std::vector<double> S = {0.501, 0.499};                               // Start probabilities
    std::vector<std::vector<double> > E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 10;      // Time Horizon
    size_t numIt = 10;  // Number of runs

    // Initial Guesses
    std::vector<std::vector<double> > AInitial{{0.61, 0.39}, {0.4, 0.6}};
    std::vector<double> SInitial{0.51, 0.49};
    std::vector<std::vector<double> > EInitial{{0.91, 0.09}, {0.1, 0.9}};

    chmmpp::HMM_inference toLearn_unconstrained(
        AInitial, SInitial, EInitial,
        0);  // 0 is the seed of the RNG, can remove and it seeds by time
    chmmpp::HMM_inference toLearn_numZeros(AInitial, SInitial, EInitial, 0);
    chmmpp::HMM_inference toLearn_stochastic(AInitial, SInitial, EInitial, 0);
    chmmpp::HMM_inference toLearn_hardEM(AInitial, SInitial, EInitial, 0);
    chmmpp::HMM_inference trueHMM(A, S, E, 0);

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<std::vector<int> > obs;
    std::vector<std::vector<int> > hid;
    std::vector<int> numZeros;
    std::vector<std::function<bool(std::vector<int>)> > constraintOracleVec;

    for (size_t i = 0; i < numIt; ++i) {
        // std::cout << "Iteration number: " << i << "\n";
        obs.push_back({});
        hid.push_back({});

        trueHMM.run(T, obs[i], hid[i]);

        auto numZerosTemp = count(hid[i].begin(), hid[i].end(), 0);
        numZeros.push_back(numZerosTemp);
        constraintOracleVec.push_back([numZerosTemp](std::vector<int> myHid) -> bool {
            return (numZerosTemp == count(myHid.begin(), myHid.end(), 0));
        });
        numZeros.push_back(count(hid[i].begin(), hid[i].end(), 0));
    }

    std::cout << "Learning without constraints.\n";
    learn_unconstrained(toLearn_unconstrained, obs);
    std::cout << "\nSoft learning with constraints.\n";
    learn_numZeros(toLearn_numZeros, obs, numZeros);
    std::cout << "\nStochastic Soft learning with constraints.\n";
    learn_stochastic(toLearn_stochastic, obs, constraintOracleVec);
    std::cout << "\nHard learning with constraints.\n";
    learn_hardEM(toLearn_hardEM, obs, constraintOracleVec, 100, 1E-6);

    std::cout << "Learned parameters without constraints:\n\n";
    toLearn_unconstrained.print();
    std::cout << "Learned parameters with constraints.\n\n";
    toLearn_numZeros.print();
    std::cout << "Stochastically learned parameters with constraints.\n\n";
    toLearn_stochastic.print();
    std::cout << "Hard learned parameters with constraints.\n\n";
    toLearn_hardEM.print();
    std::cout << "True parameter values:\n\n";
    trueHMM.print();

    return 0;
}

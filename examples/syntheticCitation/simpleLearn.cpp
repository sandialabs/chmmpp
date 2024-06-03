//
// Learn HMM parameters
//
// Generating random trials where the number of nonzeros is fixed
//
#include <iostream>
#include "syntheticCitationHMM.hpp"
template <typename T, typename V, typename Z>
void run(T& hmm, V& obs, const Z& fn)
{
    hmm.reset_rng();
    hmm.print_options();
    std::cout << std::endl;

    fn(hmm, obs);

    hmm.print();
}

bool valid(std::vector<int> hid) { //Constraint Oracle
    for(size_t t1 = 1; t1 < hid.size(); ++t1) {
        if(hid[t1] != hid[t1-1]) {
            for(size_t t2 = 0; t2 < t1-1; ++t2) {
                if(hid[t1] == hid[t2]) {
                    return false;
                }
            }
        }
    }
    return true;
}

void run_all(bool debug = false)
{
    // Initial Guess
    std::vector<std::vector<double>> A{
        {0.59, 0.11, 0.1, 0.1, 0.1}, 
        {0.09, 0.61, 0.1, 0.1, 0.1},
        {0.1, 0.1, 0.62, 0.08, 0.1}, 
        {0.1, 0.1, 0.12, 0.58, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.6}  
    };  // Transition Matrix
    std::vector<double> S = {0.2, 0.2, 0.2, 0.2, 0.2}; // Start probabilities
    std::vector<std::vector<double>> E{
        {0.6, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.6, 0.1, 0.1, 0.1},
        {0.1, 0.1, 0.6, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.6, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.6}  
    };  // Emission Matrix

    size_t T = 25;         // Time Horizon
    size_t numIt = 10;   // Number of runs

    chmmpp::HMM hmm(A, S, E, 0);

    // Store the observed and hidden variables as well as the number of zeros
    std::vector<std::vector<int>> obs(numIt);
    std::vector<std::vector<int>> hid(numIt);
    #if 0
    std::cout << "Num Obs:   " << T << std::endl;
    std::cout << "Num Runs:  " << numIt << std::endl;
    hmm.reset_rng();
    for (size_t i = 0; i < numIt; ++i) {
        while(true) {
            hmm.run(T, obs[i], hid[i]);
            if(valid(hid[i])) {
                break;
            }
        }

        if (debug) {
            std::cout << "Trial: " << i << std::endl;
            std::cout << "Observed:      ";
            for (auto& v : obs[i]) std::cout << v;
            std::cout << std::endl;

            std::cout << "Hidden states: ";
            for (auto& v : hid[i]) std::cout << v;
            std::cout << std::endl;
        }
    }

    chmmpp::HMM hmmCopy;
    chmmpp::syntheticCitationHMM schmmCopy;

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Initial HMM parameters\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmm.print();

    chmmpp::syntheticCitationHMM schmm;
    schmm.initialize(hmm);

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning without constraint - ML estimate using hidden states\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmmCopy = hmm;
    hmmCopy.estimate_hmm(obs, hid);
    hmmCopy.print();

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning without constraint - Baum-Welch\n";
    std::cout << "------------------------------------------------------------------------\n";
    hmmCopy = hmm;
    run(hmmCopy, obs,
        [](chmmpp::HMM& hmm, const std::vector<std::vector<int>>& obs) { hmm.baum_welch(obs); });

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - Soft EM\n";
    std::cout << "------------------------------------------------------------------------\n";
    schmmCopy = schmm;
    schmmCopy.set_option("C", 1000);
    //TODO This is seg faulting right 
    run(schmmCopy, obs, [](chmmpp::CHMM& hmm, const std::vector<std::vector<int>>& obs) {
        hmm.learn_stochastic(obs);
    });


    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Running learning with constraint - Hard EM\n";
    std::cout << "------------------------------------------------------------------------\n";
    schmmCopy = schmm;
    schmmCopy.set_option("max_iterations", 100000000);
    //TODO- make the number of best solutions an option
    run(schmmCopy, obs,
        [](chmmpp::CHMM& hmm, const std::vector<std::vector<int>>& obs) { hmm.learn_hardEM(obs, 100); });
    schmm.clear_options();

    std::cout << std::endl;
    #endif 
}

int main()
{
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " Generating samples\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    run_all(true);

    return 0;
}

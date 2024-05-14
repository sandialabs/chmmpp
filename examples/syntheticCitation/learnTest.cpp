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
    //hmm.print_options();
    //std::cout << std::endl;

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

//No checks
double vecError(std::vector<double> v1, std::vector<double> v2) {
    double output = 0.;

    for(size_t i = 0; i < v1.size(); ++i) {
        output = std::max(output, std::abs(v1[i]-v2[i]));
    }

    return output;
}

double matError(std::vector<std::vector<double>> A1, std::vector<std::vector<double>> A2) {
    double output = 0.;

    for(size_t i = 0; i < A1.size(); ++i) {
        for(size_t j = 0; j < A1[i].size(); ++j) {
            output = std::max(output, std::abs(A1[i][j]-A2[i][j]));
        }
    }
    
    return output;
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

    std::vector<int> numItVec = {1,10,100,1000}; 

    chmmpp::HMM hmm(A, S, E, 2);

    for(const auto &numIt: numItVec) {
        // Store the observed and hidden variables as well as the number of zeros
        std::vector<std::vector<int>> obs(numIt);
        std::vector<std::vector<int>> hid(numIt);

        std::cout << "------------------------------------------------------------------------\n";
        std::cout << "Num Runs:  " << numIt << std::endl;
        std::cout << "------------------------------------------------------------------------\n";
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

        chmmpp::HMM hmmTrue;
        chmmpp::syntheticCitationHMM schmmCopy;


        chmmpp::syntheticCitationHMM schmm;
        schmm.initialize(hmm);

        hmmTrue = hmm;
        hmmTrie.estimate_hmm(obs, hid);
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
    }
}

int main()
{
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " Generating samples\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------\n";
    run_all(false);

    return 0;
}

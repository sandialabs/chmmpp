//
// Learn HMM parameters
//
// Generating random trials where the number of nonzeros is fixed
//
#if 0
#include <iostream>
#include <chrono>
#include "syntheticCitationHMM.hpp"

template <typename T, typename V, typename Z>
int time(T& hmm, V& obs, const Z& fn)
{
    //hmm.reset_rng();
    //hmm.print_options();
    //std::cout << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    fn(hmm, obs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return static_cast<int>(duration);

    //hmm.print();
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

//perturbs the values in a vector multiplicatively by (perturb parameter)^c
//where c is uniform in [-1,1]
//Also normalizes
void perturb(std::vector<double> vec, double perturbParam) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1., 1.);

    for(auto& val: vec) val*=pow(perturbParam,dis(gen));

    //Normalize
    double sum = 0.;
    for(const auto& val: vec) sum += val;
    for(auto& val: vec) val /= sum;
}

//No checks
//l_infinity norm
double vecError(const std::vector<double> &v1, const std::vector<double> &v2) {
    double output = 0.;

    for(size_t i = 0; i < v1.size(); ++i) {
        output = std::max(output, std::abs(v1[i]-v2[i]));
    }

    return output;
}

double matError(const std::vector<std::vector<double>> &A1, const std::vector<std::vector<double>> &A2) {
    double output = 0.;

    for(size_t i = 0; i < A1.size(); ++i) {
        for(size_t j = 0; j < A1[i].size(); ++j) {
            output = std::max(output, std::abs(A1[i][j]-A2[i][j]));
        }
    }
    
    return output;
}

template <typename T>
double mean(const std::vector<T>& data) {
    double sum = 0.;
    for(const auto& val: data) sum += static_cast<double>(val);
    return sum/data.size();
}

//Standard Deviation
template <typename T>
double stdDev(const std::vector<T>& data) {
    double sum = 0.;
    double standardDeviation = 0.;

    auto mean = mean(Data);

    for (const auto& value : data) standardDeviation += pow(static_cast<double>(value) - mean, 2);

    return sqrt(standardDeviation / data.size());
}

void run_tests(bool debug = false)
{
    #if 0
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

    size_t T = 25;                                // Time Horizon
    size_t testSize = 100;                        // Number of iterations for average and stddev
    std::vector<int> numObsVec = {1,10,100,1000}; // Number of observations
    double perturbParam = 0.9;                         // How much the parameters are perturbed

    std::vector<std::vector<double>> HMM_runTimes(numObsVec.size());
    std::vector<std::vector<double>> stochastic_runTimes(numObsVec.size());
    std::vector<std::vector<double>> hardEM_runTimes(numObsVec.size());

    std::vector<std::vector<double>> HMM_error(numObsVec.size());
    std::vector<std::vector<double>> stochastic_error(numObsVec.size());
    std::vector<std::vector<double>> hardEM_error(numObsVec.size());

    chmmpp::CHMM originalCHMM;
    originalCHMM.initialize(A,S,E);
    originalCHMM.set_seed(0);

    for(size_t i = 0; i < numObsVec.size(); ++i) {
        int numObs = numObsVec[i];

        for(size_t j = 0; j < testSize; ++j) {
            std::cout << numObs << " number of observations and iteration " << j << " out of " << testSize << "\n";

            //Create Observations
            std::vector<std::vector<int>> obsVec(numObs);
            for(size_t k = 0; k < numObs; ++k) {
                std::vector<int> hid;
                std::vector<int> obs;
                originalCHMM.run(T,obs,hid);
                obsVec[k] = obs;
            }

            //Create a new CHMM where we don't exactly know the parameters
            auto perturbedA = A;
            auto perturbedS = S;
            auto perturbedE = E;

            for(auto &vec: perturbedA) {
                perturb(vec, perturbParam);
            }
            perturb(S,perturbParam);
            for(auto &vec: perturbedE) {
                perturb(vec, perturbParam);
            }
            
            //Unconstrained HMM
            std::cout << "Running unconstrained learning.\n";
            chmmpp::HMM unconstrained_HMM(perturbedA,perturbedS,perturbedE,0);
            HMM_runTimes[i].push_back(
                time(unconstrained_HMM,obsVec,
                    [](chmmpp::HMM& hmm, const std::vector<std::vector<int>>& obs) { hmm.baum_welch(obs); })
            );
            HMM_error.push_back(std::max( matError(unconstrained_HMM.getA(),A) 
                std::max(vecError(unconstrained_HMM.getS(),S), matError(unconstrained_HMM.getE(), E))
            ));

            //CHMM -- hardEM
            std::cout << "Running hardEM batch learning.\n";
            chmmpp::CHMM hardEM_CHMM;
            hardEM_CHMM.initialize(perturbedA,perturbedS, perturbedE);
            hardEM_CHMM.set_seed(0);
            hardEM_runTimes[i].push_back(
                time(hardEM_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_hardEM(obs); })
            );
            HMM_error.push_back(std::max( matError(hardEM_CHMM.hmm.getA(),A) 
                std::max(vecError(hardEM_CHMM.hmm.getS(),S), matError(unconstrained_HMM.hmm.getE(), E))
            ));

            //CHMM -- stochastic
            std::cout << "Running stochastic batch learning.\n";
            chmmpp::CHMM stochastic_CHMM;
            stochastic_CHMM.initialize(perturbedA,perturbedS, perturbedE);
            stochastic_CHMM.set_seed(0);
            stochastic_runTimes[i].push_back(
                time(stochastic_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_stochastic(obs); })
            );
            HMM_error.push_back(std::max( matError(stochastic_CHMM.hmm.getA(),A) 
                std::max(vecError(stochastic_CHMM.hmm.getS(),S), matError(unconstrained_HMM.hmm.getE(), E))
            )); 
        }
    }

    for(size_t i = 0; i < numObsVec.size(); ++i) {
        auto numObs = numObsVec[i];
        std::cout << "Printing statistics with " << numObs << " observations.\n";
        std::cout << "----------------------------------------------------------\n\n";

        
    }
    #endif

}
#endif
int main()
{

    //run_tests(false);

    return 0;
}

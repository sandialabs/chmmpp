//
// Learn HMM parameters
//
// Generating random trials with citation constraint
//

#include <iostream>
#include <chrono>
#include "syntheticCitationHMM.hpp"

//Output is in milliseconds
template <typename T, typename V, typename Z>
size_t time(T& hmm, V& obs, const Z& fn)
{
    //hmm.reset_rng();
    //hmm.print_options();
    //std::cout << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    fn(hmm, obs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return static_cast<size_t>(duration);

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
void perturb(std::vector<double>& vec, const double& perturbParam) {
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

    auto my_mean = mean(data);

    for (const auto& value : data) standardDeviation += pow(static_cast<double>(value) - my_mean, 2);

    return sqrt(standardDeviation / data.size());
}

void run_tests(bool debug = false)
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

    size_t T = 25;                                // Time Horizon
    size_t testSize = 100;                        // Number of iterations for average and stddev
    //std::vector<int> numObsVec = {1,10,100,1000}; // Number of observations
    std::vector<int> numObsVec = {10};
    double perturbParam = 0.9;                         // How much the parameters are perturbed

    std::vector<std::vector<size_t>> HMM_run_times(numObsVec.size());
    std::vector<std::vector<size_t>> stochastic_run_times(numObsVec.size());
    std::vector<std::vector<size_t>> hardEM_run_times(numObsVec.size());

    std::vector<std::vector<double>> HMM_error(numObsVec.size());
    std::vector<std::vector<double>> stochastic_error(numObsVec.size());
    std::vector<std::vector<double>> hardEM_error(numObsVec.size());

    std::vector<std::vector<double>> HMM_log_likelihood(numObsVec.size());
    std::vector<std::vector<double>> stochastic_log_likelihood(numObsVec.size());
    std::vector<std::vector<double>> hardEM_log_likelihood(numObsVec.size());

    chmmpp::syntheticCitationHMM originalCHMM;
    originalCHMM.initialize(A,S,E);
    originalCHMM.set_seed(0);

    for(size_t i = 0; i < numObsVec.size(); ++i) {
        int numObs = numObsVec[i];

        HMM_run_times.push_back({});
        stochastic_run_times.push_back({});
        hardEM_run_times.push_back({});

        HMM_error.push_back({});
        stochastic_error.push_back({});
        hardEM_error.push_back({});

        HMM_log_likelihood.push_back({});
        stochastic_log_likelihood.push_back({});
        hardEM_log_likelihood.push_back({});

        for(size_t j = 0; j < testSize; ++j) {
            std::cout << numObs << " observations and iteration " << j << " out of " << testSize << "\n";

            //Create Observations
            std::vector<std::vector<int>> obsVec(numObs);
            std::vector<std::vector<int>> hidVec(numObs);
            for(size_t k = 0; k < numObs; ++k) {
                std::vector<int> hid;
                std::vector<int> obs;
                originalCHMM.run(T,obs,hid);
                obsVec[k] = obs;
                hidVec[k] = hid;
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

            chmmpp::HMM perturbed_HMM(perturbedA,perturbedS,perturbedE,0);

            if(debug) {
                std::cout << "Perturbed parameters:\n";
                perturbed_HMM.print();
                std::cout << "Observations:\n";
                for(size_t i = 0; i < obsVec.size(); ++i) {
                    for(size_t j = 0; j < obsVec[i].size(); ++j) {
                        std::cout << obsVec[i][j];
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
                std::cout << "Hidden:\n";
                for(size_t i = 0; i < hidVec.size(); ++i) {
                    for(size_t j = 0; j < hidVec[i].size(); ++j) {
                        std::cout << hidVec[i][j];
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            
            //Unconstrained HMM
            std::cout << "Running unconstrained learning.\n";
            chmmpp::HMM unconstrained_HMM = perturbed_HMM;
            unconstrained_HMM.set_seed(1);
            HMM_run_times[i].push_back(
                time(unconstrained_HMM,obsVec,
                    [](chmmpp::HMM& hmm, const std::vector<std::vector<int>>& obs) { hmm.baum_welch(obs); })
            );
            HMM_error[i].push_back(std::max( matError(unconstrained_HMM.getA(),A), 
                std::max(vecError(unconstrained_HMM.getS(),S), matError(unconstrained_HMM.getE(), E))
            ));
            chmmpp::syntheticCitationHMM unconstrained_CHMM;
            unconstrained_CHMM.initialize(unconstrained_HMM);
            HMM_log_likelihood[i].push_back(unconstrained_CHMM.log_likelihood_estimate(obsVec));           

            //CHMM -- hardEM
            std::cout << "Running hardEM batch learning.\n";
            chmmpp::syntheticCitationHMM hardEM_CHMM;
            hardEM_CHMM.initialize(chmmpp::HMM(perturbed_HMM));
            hardEM_CHMM.set_seed(0);
            hardEM_run_times[i].push_back(
                time(hardEM_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_hardEM_constraint_oracle(obs); })
            );
            hardEM_error[i].push_back(std::max( matError(hardEM_CHMM.hmm.getA(),A),
                std::max(vecError(hardEM_CHMM.hmm.getS(),S), matError(hardEM_CHMM.hmm.getE(), E))
            ));
            hardEM_log_likelihood[i].push_back(hardEM_CHMM.log_likelihood_estimate(obsVec));

            //CHMM -- stochastic
            std::cout << "Running stochastic batch learning.\n";
            chmmpp::syntheticCitationHMM stochastic_CHMM;
            stochastic_CHMM.initialize(chmmpp::HMM(perturbed_HMM));
            stochastic_CHMM.set_seed(0);
            stochastic_run_times[i].push_back(
                time(stochastic_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_stochastic_constraint_oracle(obs); })
            );
            stochastic_error[i].push_back(std::max( matError(stochastic_CHMM.hmm.getA(),A),
                std::max(vecError(stochastic_CHMM.hmm.getS(),S), matError(stochastic_CHMM.hmm.getE(), E))
            ));
            stochastic_log_likelihood[i].push_back(stochastic_CHMM.log_likelihood_estimate(obsVec));
            std::cout << "\n"; 
        }
    }

    for(size_t i = 0; i < numObsVec.size(); ++i) {
        auto numObs = numObsVec[i];
        std::cout << "Printing statistics with " << numObs << " observations.\n";
        std::cout << "-------------------------------------------------------\n\n";

        std::cout << "Average HMM error: " << mean(HMM_error[i]) << "\n";
        std::cout << "HMM error standard deviation: " << stdDev(HMM_error[i]) << "\n";
        std::cout << "Average HMM running time: " << mean(HMM_run_times[i]) << "\n";
        std::cout << "HMM running time standard deviation: " << stdDev(HMM_run_times[i]) << "\n";
        std::cout << "Average HMM log likelihood: " << mean(HMM_log_likelihood[i]) << "\n";
        std::cout << "HMM log likelihood standard deviation: " << stdDev(HMM_log_likelihood[i]) << "\n\n";
        
        std::cout << "Average stochastic error: " << mean(stochastic_error[i]) << "\n";
        std::cout << "stochastic error standard deviation: " << stdDev(stochastic_error[i]) << "\n";
        std::cout << "Average stochastic running time: " << mean(stochastic_run_times[i]) << "\n";
        std::cout << "stochastic running time standard deviation: " << stdDev(stochastic_run_times[i]) << "\n";
        std::cout << "Average stochastic log likelihood: " << mean(stochastic_log_likelihood[i]) << "\n";
        std::cout << "stochastic log likelihood standard deviation: " << stdDev(stochastic_log_likelihood[i]) << "\n\n";

        std::cout << "Average hardEM error: " << mean(hardEM_error[i]) << "\n";
        std::cout << "hardEM error standard deviation: " << stdDev(hardEM_error[i]) << "\n";
        std::cout << "Average hardEM running time: " << mean(hardEM_run_times[i]) << "\n";
        std::cout << "hardEM running time standard deviation: " << stdDev(hardEM_run_times[i]) << "\n";
        std::cout << "Average hardEM log likelihood: " << mean(hardEM_log_likelihood[i]) << "\n";
        std::cout << "hardEM log likelihood standard deviation: " << stdDev(hardEM_log_likelihood[i]) << "\n\n\n";
    }

}

int main()
{

    run_tests(false);

    return 0;
}

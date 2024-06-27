//
// Learn HMM parameters
//
// Generating random trials with citation constraint
//

#include <iostream>
#include <fstream>
#include <chrono>
#include "numZerosHMM.hpp"

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
        output = std::max(output, std::fabs(v1[i]-v2[i]));
    }

    return output;
}

double matError(const std::vector<std::vector<double>> &A1, const std::vector<std::vector<double>> &A2) {
    double output = 0.;

    for(size_t i = 0; i < A1.size(); ++i) {
        for(size_t j = 0; j < A1[i].size(); ++j) {
            output = std::max(output, std::fabs(A1[i][j]-A2[i][j]));
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

//Validation Error
//This is not as trivial as it seems
//The issue is that we could possibly permute the hidden states and still get the same probability distribution of observations
//This wouldn't be an issue if we were doing semi-supervised learning I bet
//This is not the best coding, there should be a way to do constrained and unconstrained at the same time
double validation_error(chmmpp::numZerosHMM& nzhmm, const std::vector<std::vector<int>>& obsVec, const std::vector<std::vector<int>>& hidVec) {
    size_t H = nzhmm.hmm.getH();
    
    std::vector<size_t> perm(H);
    for(size_t h = 0; h < H; ++h) {
        perm[h] = h;
    }

    size_t num_transistions = 0.;
    for(const auto &vec: obsVec) {
        num_transistions += vec.size(); 
    }
    auto min_num_errors = num_transistions;

    std::vector<std::vector<int>> hidVecGuess(obsVec.size());
    for(size_t n = 0; n < obsVec.size(); ++n) {
        double temp;
        nzhmm.aStar(obsVec[n],hidVecGuess[n],temp);
    }

    do {
        size_t num_errors = 0;
        for(size_t n = 0; n < hidVec.size(); ++n) {
            for(size_t t = 0; t < hidVec[n].size(); ++t) {
                if(hidVec[n][t] != perm[hidVecGuess[n][t]]) {
                    ++num_errors;
                }
            }
        }
        min_num_errors = std::min(num_errors, min_num_errors);
    } while (std::next_permutation(perm.begin(), perm.end()));

    return ((double) min_num_errors)/((double) num_transistions);
} 

void run_tests(bool debug = false)
{

    // Initial Guess
    std::vector<std::vector<double>> A{{0.6, 0.4}, {0.3, 0.7}};  // Transition Matrix
    std::vector<double> S = {0.4, 0.6};                          // Start probabilities
    std::vector<std::vector<double>> E{{0.699, 0.301}, {0.299, 0.701}};  // Emission Matrix

    size_t T = 25;                                // Time Horizon
    size_t numZeros = 10;                              // Number of zeros
    size_t testSize = 100;                        // Number of iterations for average and stddev
    //std::vector<int> numObsVec = {1,10,100,1000}; // Number of observations
    std::vector<int> numObsVec = {1};
    size_t num_valid = 40;                         // Number of sets of observations for validation
                                                  // Chose to give 1000 time steps
    double perturbParam = 0.9;                         // How much the parameters are perturbed

    std::vector<std::vector<size_t>> HMM_run_times(numObsVec.size());
    std::vector<std::vector<size_t>> unconstrained_run_times(numObsVec.size());
    std::vector<std::vector<size_t>> stochastic_run_times(numObsVec.size());
    std::vector<std::vector<size_t>> hardEM_run_times(numObsVec.size());
    std::vector<std::vector<size_t>> MIP_run_times(numObsVec.size());

    std::vector<std::vector<double>> HMM_error(numObsVec.size());
    std::vector<std::vector<double>> unconstrained_error(numObsVec.size());
    std::vector<std::vector<double>> stochastic_error(numObsVec.size());
    std::vector<std::vector<double>> hardEM_error(numObsVec.size());
    std::vector<std::vector<double>> MIP_error(numObsVec.size());

    std::vector<std::vector<double>> HMM_log_likelihood(numObsVec.size());
    std::vector<std::vector<double>> unconstrained_log_likelihood(numObsVec.size());
    std::vector<std::vector<double>> stochastic_log_likelihood(numObsVec.size());
    std::vector<std::vector<double>> hardEM_log_likelihood(numObsVec.size());
    std::vector<std::vector<double>> MIP_log_likelihood(numObsVec.size());

    std::vector<std::vector<double>> HMM_validation_error(numObsVec.size());
    std::vector<std::vector<double>> unconstrained_validation_error(numObsVec.size());
    std::vector<std::vector<double>> stochastic_validation_error(numObsVec.size());
    std::vector<std::vector<double>> hardEM_validation_error(numObsVec.size());
    std::vector<std::vector<double>> MIP_validation_error(numObsVec.size());

    chmmpp::numZerosHMM originalCHMM(numZeros);
    originalCHMM.initialize(A,S,E);
    originalCHMM.set_seed(0);

    for(size_t i = 0; i < numObsVec.size(); ++i) {
        int numObs = numObsVec[i];

        for(size_t j = 0; j < testSize; ++j) {
            std::cout << numObs << " observations and iteration " << j << " out of " << testSize << "\n";

            //Create Observations
            std::vector<std::vector<int>> obsVec(numObs);
            std::vector<std::vector<int>> hidVec(numObs);
            for(size_t k = 0; k < numObs; ++k) {
                std::vector<int> hid(T);
                std::vector<int> obs(T);
                originalCHMM.run(obs,hid);
                obsVec[k] = obs;
                hidVec[k] = hid;
            }

            //Creating Validation Data            
            std::vector<std::vector<int>> obsVec_validation(num_valid);
            std::vector<std::vector<int>> hidVec_validation(num_valid);
            for(size_t k = 0; k < num_valid; ++k) {
                std::vector<int> hid(T);
                std::vector<int> obs(T);
                originalCHMM.run(obs,hid);
                obsVec_validation[k] = obs;
                hidVec_validation[k] = hid;
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

            chmmpp::HMM trueHMM;
            trueHMM.initialize(A,S,E);
            trueHMM.estimate_hmm(obsVec,hidVec);
            auto trueA = trueHMM.getA();
            auto trueS = trueHMM.getS();
            auto trueE = trueHMM.getE();
            
            //Unconstrained HMM
            std::cout << "Running unconstrained learning." << std::endl;
            chmmpp::HMM unconstrained_HMM = perturbed_HMM;
            unconstrained_HMM.set_seed(1);
            HMM_run_times[i].push_back(
                time(unconstrained_HMM,obsVec,
                    [](chmmpp::HMM& hmm, const std::vector<std::vector<int>>& obs) { hmm.baum_welch(obs); })
            );
            HMM_error[i].push_back(std::max( matError(unconstrained_HMM.getA(),trueA), 
                std::max(vecError(unconstrained_HMM.getS(),trueS), matError(unconstrained_HMM.getE(), trueE))
            ));
            chmmpp::numZerosHMM temp_CHMM(numZeros);
            temp_CHMM.initialize(unconstrained_HMM);
            HMM_log_likelihood[i].push_back(temp_CHMM.log_likelihood_estimate(obsVec));    
            HMM_validation_error[i].push_back(validation_error(temp_CHMM,obsVec_validation,hidVec_validation));           
       

            //CHMM -- unconstrained
            std::cout << "Running unconstrained batch learning.\n";
            chmmpp::numZerosHMM unconstrained_CHMM(numZeros);
            unconstrained_CHMM.initialize(chmmpp::HMM(perturbed_HMM));
            unconstrained_CHMM.set_seed(0);
            unconstrained_run_times[i].push_back(
                time(unconstrained_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_unconstrained(obs); })
            );
            unconstrained_error[i].push_back(std::max( matError(unconstrained_CHMM.hmm.getA(),trueA),
                std::max(vecError(unconstrained_CHMM.hmm.getS(),trueS), matError(unconstrained_CHMM.hmm.getE(), trueE))
            ));
            unconstrained_log_likelihood[i].push_back(unconstrained_CHMM.log_likelihood_estimate(obsVec));
            unconstrained_validation_error[i].push_back(validation_error(unconstrained_CHMM,obsVec_validation,hidVec_validation));  

            //CHMM -- hardEM
            std::cout << "Running hardEM batch learning.\n";
            chmmpp::numZerosHMM hardEM_CHMM(numZeros);
            hardEM_CHMM.initialize(chmmpp::HMM(perturbed_HMM));
            hardEM_CHMM.set_seed(0);
            hardEM_run_times[i].push_back(
                time(hardEM_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_hardEM_constraint_oracle(obs); })
            );
            hardEM_error[i].push_back(std::max( matError(hardEM_CHMM.hmm.getA(),trueA),
                std::max(vecError(hardEM_CHMM.hmm.getS(),trueS), matError(hardEM_CHMM.hmm.getE(), trueE))
            ));
            hardEM_log_likelihood[i].push_back(hardEM_CHMM.log_likelihood_estimate(obsVec));
            hardEM_validation_error[i].push_back(validation_error(hardEM_CHMM,obsVec_validation,hidVec_validation));  

            //CHMM -- stochastic
            std::cout << "Running stochastic batch learning.\n";
            chmmpp::numZerosHMM stochastic_CHMM(numZeros);
            stochastic_CHMM.initialize(chmmpp::HMM(perturbed_HMM));
            stochastic_CHMM.set_seed(0);
            stochastic_run_times[i].push_back(
                time(stochastic_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_stochastic_constraint_oracle(obs); })
            );
            stochastic_error[i].push_back(std::max( matError(stochastic_CHMM.hmm.getA(),trueA),
                std::max(vecError(stochastic_CHMM.hmm.getS(),trueS), matError(stochastic_CHMM.hmm.getE(), trueE))
            ));
            stochastic_log_likelihood[i].push_back(stochastic_CHMM.log_likelihood_estimate(obsVec));
            stochastic_validation_error[i].push_back(validation_error(stochastic_CHMM,obsVec_validation,hidVec_validation));

            //CHMM -- MIP
            std::cout << "Running MIP batch learning.\n";
            chmmpp::numZerosHMM MIP_CHMM(numZeros);
            MIP_CHMM.initialize(chmmpp::HMM(perturbed_HMM));
            MIP_CHMM.set_seed(0);
            MIP_run_times[i].push_back(
                time(MIP_CHMM,obsVec,
                    [](chmmpp::CHMM& chmm, const std::vector<std::vector<int>>& obs) { chmm.learn_MIP_generator(obs); })
            );
            MIP_error[i].push_back(std::max( matError(MIP_CHMM.hmm.getA(),A),
                std::max(vecError(MIP_CHMM.hmm.getS(),S), matError(MIP_CHMM.hmm.getE(), E))
            ));
            MIP_log_likelihood[i].push_back(MIP_CHMM.log_likelihood_estimate(obsVec));
            MIP_validation_error[i].push_back(validation_error(MIP_CHMM,obsVec_validation,hidVec_validation));

            std::cout << "\n"; 
        }
    }

        for(size_t i = 0; i < numObsVec.size(); ++i) {
        std::ofstream outputFile("syntheticCitation_learn_comparison_"+std::to_string(numObsVec[i])+".txt");

        auto numObs = numObsVec[i];
        std::cout << "Printing statistics with " << numObs << " observations.\n";
        std::cout << "-------------------------------------------------------\n\n";

        std::cout << "Average HMM error: " << mean(HMM_error[i]) << "\n";
        std::cout << "HMM error standard deviation: " << stdDev(HMM_error[i]) << "\n";
        std::cout << "Average HMM running time: " << mean(HMM_run_times[i]) << "\n";
        std::cout << "HMM running time standard deviation: " << stdDev(HMM_run_times[i]) << "\n";
        std::cout << "Average HMM log likelihood: " << mean(HMM_log_likelihood[i]) << "\n";
        std::cout << "HMM log likelihood standard deviation: " << stdDev(HMM_log_likelihood[i]) << "\n";
        std::cout << "Average HMM validation error: " << mean(HMM_validation_error[i]) << "\n";
        std::cout << "HMM validation error standard deviation: " << stdDev(HMM_validation_error[i]) << "\n\n";
    
        std::cout << "Average unconstrained error: " << mean(unconstrained_error[i]) << "\n";
        std::cout << "unconstrained error standard deviation: " << stdDev(unconstrained_error[i]) << "\n";
        std::cout << "Average unconstrained running time: " << mean(unconstrained_run_times[i]) << "\n";
        std::cout << "unconstrained running time standard deviation: " << stdDev(unconstrained_run_times[i]) << "\n";
        std::cout << "Average unconstrained log likelihood: " << mean(unconstrained_log_likelihood[i]) << "\n";
        std::cout << "unconstrained log likelihood standard deviation: " << stdDev(unconstrained_log_likelihood[i]) << "\n";
        std::cout << "Average unconstrained validation error: " << mean(unconstrained_validation_error[i]) << "\n";
        std::cout << "unconstrained validation error standard deviation: " << stdDev(unconstrained_validation_error[i]) << "\n\n";

        std::cout << "Average stochastic error: " << mean(stochastic_error[i]) << "\n";
        std::cout << "stochastic error standard deviation: " << stdDev(stochastic_error[i]) << "\n";
        std::cout << "Average stochastic running time: " << mean(stochastic_run_times[i]) << "\n";
        std::cout << "stochastic running time standard deviation: " << stdDev(stochastic_run_times[i]) << "\n";
        std::cout << "Average stochastic log likelihood: " << mean(stochastic_log_likelihood[i]) << "\n";
        std::cout << "stochastic log likelihood standard deviation: " << stdDev(stochastic_log_likelihood[i]) << "\n";
        std::cout << "Average stochastic validation error: " << mean(stochastic_validation_error[i]) << "\n";
        std::cout << "stochastic validation error standard deviation: " << stdDev(stochastic_validation_error[i]) << "\n\n";

        std::cout << "Average hardEM error: " << mean(hardEM_error[i]) << "\n";
        std::cout << "hardEM error standard deviation: " << stdDev(hardEM_error[i]) << "\n";
        std::cout << "Average hardEM running time: " << mean(hardEM_run_times[i]) << "\n";
        std::cout << "hardEM running time standard deviation: " << stdDev(hardEM_run_times[i]) << "\n";
        std::cout << "Average hardEM log likelihood: " << mean(hardEM_log_likelihood[i]) << "\n";
        std::cout << "hardEM log likelihood standard deviation: " << stdDev(hardEM_log_likelihood[i]) << "\n";
        std::cout << "Average hardEM validation error: " << mean(hardEM_validation_error[i]) << "\n";
        std::cout << "hardEM validation error standard deviation: " << stdDev(hardEM_validation_error[i]) << "\n\n";
        
        std::cout << "Average MIP error: " << mean(MIP_error[i]) << "\n";
        std::cout << "MIP error standard deviation: " << stdDev(MIP_error[i]) << "\n";
        std::cout << "Average MIP running time: " << mean(MIP_run_times[i]) << "\n";
        std::cout << "MIP running time standard deviation: " << stdDev(MIP_run_times[i]) << "\n";
        std::cout << "Average MIP log likelihood: " << mean(MIP_log_likelihood[i]) << "\n";
        std::cout << "MIP log likelihood standard deviation: " << stdDev(MIP_log_likelihood[i]) << "\n";
        std::cout << "Average MIP validation error: " << mean(MIP_validation_error[i]) << "\n";
        std::cout << "MIP validation error standard deviation: " << stdDev(MIP_validation_error[i]) << "\n\n\n";
        

        outputFile << "Statistics with " << numObs << " observations.\n";
        outputFile << "-------------------------------------------------------\n\n";

        outputFile << "Average HMM error: " << mean(HMM_error[i]) << "\n";
        outputFile << "HMM error standard deviation: " << stdDev(HMM_error[i]) << "\n";
        outputFile << "Average HMM running time: " << mean(HMM_run_times[i]) << "\n";
        outputFile << "HMM running time standard deviation: " << stdDev(HMM_run_times[i]) << "\n";
        outputFile << "Average HMM log likelihood: " << mean(HMM_log_likelihood[i]) << "\n";
        outputFile << "HMM log likelihood standard deviation: " << stdDev(HMM_log_likelihood[i]) << "\n";
        outputFile << "Average HMM validation error: " << mean(HMM_validation_error[i]) << "\n";
        outputFile << "HMM validation error standard deviation: " << stdDev(HMM_validation_error[i]) << "\n\n";

        outputFile << "Average unconstrained error: " << mean(unconstrained_error[i]) << "\n";
        outputFile << "unconstrained error standard deviation: " << stdDev(unconstrained_error[i]) << "\n";
        outputFile << "Average unconstrained running time: " << mean(unconstrained_run_times[i]) << "\n";
        outputFile << "unconstrained running time standard deviation: " << stdDev(unconstrained_run_times[i]) << "\n";
        outputFile << "Average unconstrained log likelihood: " << mean(unconstrained_log_likelihood[i]) << "\n";
        outputFile << "unconstrained log likelihood standard deviation: " << stdDev(unconstrained_log_likelihood[i]) << "\n";
        outputFile << "Average unconstrained validation error: " << mean(unconstrained_validation_error[i]) << "\n";
        outputFile << "unconstrained validation error standard deviation: " << stdDev(unconstrained_validation_error[i]) << "\n\n";

        outputFile << "Average stochastic error: " << mean(stochastic_error[i]) << "\n";
        outputFile << "stochastic error standard deviation: " << stdDev(stochastic_error[i]) << "\n";
        outputFile << "Average stochastic running time: " << mean(stochastic_run_times[i]) << "\n";
        outputFile << "stochastic running time standard deviation: " << stdDev(stochastic_run_times[i]) << "\n";
        outputFile << "Average stochastic log likelihood: " << mean(stochastic_log_likelihood[i]) << "\n";
        outputFile << "stochastic log likelihood standard deviation: " << stdDev(stochastic_log_likelihood[i]) << "\n";
        outputFile << "Average stochastic validation error: " << mean(stochastic_validation_error[i]) << "\n";
        outputFile << "stochastic validation error standard deviation: " << stdDev(stochastic_validation_error[i]) << "\n\n";

        outputFile << "Average hardEM error: " << mean(hardEM_error[i]) << "\n";
        outputFile << "hardEM error standard deviation: " << stdDev(hardEM_error[i]) << "\n";
        outputFile << "Average hardEM running time: " << mean(hardEM_run_times[i]) << "\n";
        outputFile << "hardEM running time standard deviation: " << stdDev(hardEM_run_times[i]) << "\n";
        outputFile << "Average hardEM log likelihood: " << mean(hardEM_log_likelihood[i]) << "\n";
        outputFile << "hardEM log likelihood standard deviation: " << stdDev(hardEM_log_likelihood[i]) << "\n";
        outputFile << "Average hardEM validation error: " << mean(hardEM_validation_error[i]) << "\n";
        outputFile << "hardEM validation error standard deviation: " << stdDev(hardEM_validation_error[i]) << "\n\n";
    
        outputFile << "Average MIP error: " << mean(MIP_error[i]) << "\n";
        outputFile << "MIP error standard deviation: " << stdDev(MIP_error[i]) << "\n";
        outputFile << "Average MIP running time: " << mean(MIP_run_times[i]) << "\n";
        outputFile << "MIP running time standard deviation: " << stdDev(MIP_run_times[i]) << "\n";
        outputFile << "Average MIP log likelihood: " << mean(MIP_log_likelihood[i]) << "\n";
        outputFile << "MIP log likelihood standard deviation: " << stdDev(MIP_log_likelihood[i]) << "\n";
        outputFile << "Average MIP validation error: " << mean(MIP_validation_error[i]) << "\n";
        outputFile << "MIP validation error standard deviation: " << stdDev(MIP_validation_error[i]) << "\n\n\n";
    }
}

int main()
{

    run_tests(false);

    return 0;
}

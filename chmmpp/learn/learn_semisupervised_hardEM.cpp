#include <iostream>
#include "learn.hpp"
#include "../inference/inference.hpp"

namespace chmmpp {

//CLM - TODO -- I don't fully understand this part yet, I just need a parameter called gamma which 
//    dictates how heavily the unsupervised samples are underweighted
void process_options(const Options &options, double &convergence_tolerance, unsigned int &C, double &gamma)
{
    for (const auto &it : options.options) {
        if (it.first == "C") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp > 0)
                    C = tmp;
                else
                    std::cerr << "WARNING: 'C' option must be a non-negative integer" << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second)) {
                C = std::get<unsigned int>(it.second);
            }
            else
                std::cerr << "WARNING: 'C' option must be a non-negative integer" << std::endl;
        }
        else if (it.first == "convergence_tolerance") {
            if (std::holds_alternative<double>(it.second))
                convergence_tolerance = std::get<double>(it.second);
            else
                std::cerr << "WARNING: 'convergence_tolerance' option must be a double"
                          << std::endl;
        }
    }
}

template<typename T>
void normalize(std::vector<T> &vec) {
    double sum = 0.;
    for(const auto &val: vec)
        sum += val;

    if(sum != 0) {
        for(auto &val: vec)
            val /= sum;
    }
    else {
        double vecSizeInv = 1./vec.size();
        for(auto &val: vec)
            val = vecSizeInv;
    }
}

void learn_semisupervised_hardEM(HMM &hmm, const std::vector< std::vector<int> > &supervisedObs, 
                           const std::vector< std::vector<int> > &supervisedHidden,
                           const std::vector< std::vector<int> > &unsupervisedObs,
                           const std::function<bool(std::vector<int>&)> &constraintOracle,
                           bool partialOracle, const Options &options) 
{    
    double gamma = 0.1; //Does it make sense to just have a double or should it depend on the size of
                        //supervised vs. unsupervised data
    double convergence_tolerance = 1E-6;

    size_t H = hmm.getH();
    size_t O = hmm.getO();

    //Idea: obs and hidden will contain all observations and hidden across sup and unsup
    //Hidden for unsup will be generated using inference
    //Then do normal hardEM BUT weight unsup examples by gamma
    std::vector< std::vector<int> > obs;
    std::vector< std::vector<int> > hidden;
    int supervisedSize = supervisedObs.size();

    for(const auto &vec: supervisedObs)
        obs.push_back(vec);

    for(const auto &vec: unsupervisedObs) 
        obs.push_back(vec);

    for(const auto &vec: supervisedHidden) 
        hidden.push_back(vec);

    HMM hmmCopy = hmm;

    //Initialize the matrix better than just uniform using the supervised training data
    estimate_hmm(hmmCopy, supervisedObs, supervisedHidden);
    std::vector< std::vector<double> > newA = hmmCopy.getA();
    std::vector<double> newS = hmmCopy.getS();
    std::vector< std::vector<double> > newE = hmmCopy.getE();

    for(size_t h = 0; h < H; ++h) {
        
        newS[h] += gamma*hmm.getSEntry(h);
        for(size_t h2 = 0; h2 < H; ++h2) {
            newA[h][h2] += gamma*hmm.getAEntry(h,h2);
        }
        for(size_t o = 0; o < O; ++o) {
            newE[h][o] += gamma*hmm.getEEntry(h,o);
        }
    }
    
    for(auto &vec: newA)
        normalize(vec);

    normalize(newS);

    for(auto &vec: newE) 
        normalize(vec);

    hmm.initialize(newA,newS,newE);

    //Iteratively 
    while(true) {
        std::cout << "Running learning." << std::endl;
        hidden.clear();
        for(const auto &vec: supervisedHidden) 
            hidden.push_back(vec);

        //Hard part of hard EM
        int temp = 1;
        for(const auto &vec: unsupervisedObs) { 
            std::vector<int> tempHidden;
            double tempLogProb;
            aStarOracle(hmm, vec, tempHidden, tempLogProb, constraintOracle, partialOracle); //Could replace with IP or specific aStar
            hidden.push_back(tempHidden);
            std::cout << "Iteration " << temp << " out of " << unsupervisedObs.size() << "\n";
            ++temp;
        }
        
        std::vector<std::vector<double> > ACounter(H); //double not int b/c we multiply by gamma
        std::vector<double> SCounter(H);
        std::vector<std::vector<double> > ECounter(H);

        for(auto &vec: ACounter)
            vec.resize(H);
        for(auto &vec: ECounter)
            vec.resize(O);

        size_t counter = 0;
        //Do counts weighting unsupervised samples less
        for(const auto &hiddenVec: hidden) {
            size_t T = hiddenVec.size();
            double incrementVal;

            if(counter < supervisedSize)
                incrementVal = 1;
            else
                incrementVal = gamma;

            SCounter[hiddenVec[0]] += incrementVal;
            ECounter[hiddenVec[T-1]][obs[counter][T-1]] += incrementVal;

            for(int t = 0; t < T-1; ++t) {
                ACounter[hiddenVec[t]][hiddenVec[t+1]] += incrementVal;
                ECounter[hiddenVec[t]][obs[counter][t]] += incrementVal;
            }

            ++counter;
        }

        for(auto &vec: ACounter)
            normalize(vec);
            
        normalize(SCounter);

        for(auto &vec: ECounter)
            normalize(vec);

        double tol = -1.;
        for(int h1 = 0; h1 < H; ++h1) {
            for(int h2 = 0; h2 < H; ++h2) {
                tol = std::max(std::abs(hmm.getAEntry(h1,h2) - ACounter[h1][h2]), tol);
            }
        }

        hmm.initialize(ACounter, SCounter,ECounter);
        hmm.print();
        std::cout << tol << std::endl << std::endl << std::endl;

        if(tol < convergence_tolerance)
            break;
    }
}

}  // namespace chmmpp

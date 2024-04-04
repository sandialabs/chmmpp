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
    double vecSizeInv = 1./vec.size();

    for(const auto &val: vec) {
        sum += val;
    }

    if(sum != 0) {
        for(auto &val: vec) {
            val /= sum;
        }
    }
    else {
        for(auto &val: vec) {
            val = vecSizeInv;
        }
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

    int H = hmm.getH();
    int O = hmm.getO();

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
        
        newS[h] += hmm.getSEntry(h);
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
    hmm.print();

    for(const auto &vec: unsupervisedObs) { 
        std::vector<int> tempHidden;
        double tempLogProb;
        aStarOracle(hmm, vec, tempHidden, tempLogProb, constraintOracle, partialOracle);
        //hmm.viterbi(vec,tempHidden,tempLogProb);
        for(const auto &val: tempHidden) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        hidden.push_back(tempHidden);
    }
}

}  // namespace chmmpp

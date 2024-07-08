#include <iostream>
#include "learn.hpp"
#include "../inference/inference.hpp"
#include <iomanip>

namespace {
void normalizeEps(std::vector<double> &myVec) {
    double sum = 0.;
    double eps = 1E-7;

    for(const auto &elem: myVec) sum += elem;

    if(sum != 0.) {
        for(auto &elem: myVec) elem /= sum;
    }
    else {
        for(auto &elem: myVec) elem = eps;
    }
}

void normalize(std::vector<double> &myVec) {
    double eps = 1E-7;
    double sum = 0.;
    for(auto &elem: myVec) { //We can't have 0 probabilities in S for the MIP because otherwise we may project into infeasible solutions
        if(std::fabs(elem) < eps) {
            elem = eps;
        }
    }

    for(const auto &elem: myVec) sum += elem;

    if(sum != 0.) {
        for(auto &elem: myVec) elem /= sum;
    }
    else {
        for(auto &elem: myVec) elem = 1./((double) myVec.size());
    }
}

}

namespace chmmpp {

//Each itertation of generated hidden states are underweighted by 1, 1/2, 1/3, ...
void learn_batch(HMM &hmm, 
                const std::vector<std::vector<int>> &obs, 
                const Generator_Base &generator,
                const Options& options) 
{
    //TODO Make into options
    const double convergence_tolerance = 10E-6;
    const unsigned int max_iteration = 1000000;
    const unsigned int max_iteration_generator = 0;
    //TODO this doesn't work right now const int num_solutions = 100; //Assumes constant number of solutions each time 
    const double convergeFactor = -1.;

    std::vector<std::vector<std::vector<std::vector<double>>>> gamma;
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> xi; 

    const int H = hmm.getH();
    const int O = hmm.getO();
    const int R = obs.size();

    int numIt = 0;

    std::vector<std::vector<double>> newA(H);
    std::vector<std::vector<double>> newE(H);
    std::vector<double> newS(H,0.);
    for(auto &vec: newA) vec.resize(H,0.);
    for(auto &vec: newE) vec.resize(O,0.);

    while(true) {
        ++numIt;//==gamma.size();
        auto newHidden = generator(hmm, obs); //r,n,t
        //std::cout << "TEST " << obs.size() << " " << obs[0].size() << std::endl;

        /*for(size_t r = 0; r < obs.size(); ++r) {
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                for(size_t t = 0; t < newHidden[r][n].size(); ++t) {
                    std::cout << newHidden[r][n][t];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }*/

        std::vector<std::vector<double>> newProbs(obs.size()); //r,n
        for(size_t r = 0; r < obs.size(); ++r) { 
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                newProbs[r].push_back(hmm.logProb(obs[r], newHidden[r][n]));
            }
        }

        //Generate xi and gamma
        std::vector<std::vector<std::vector<double>>> gamma(R); //r, t, h
        std::vector<std::vector<std::vector<std::vector<double>>>> xi(R); //r, t, h, g

        for(size_t r = 0 ; r < R; ++r) {
            gamma[r].resize(obs[r].size());
            for(size_t t = 0; t < obs[r].size(); ++t) {
                gamma[r][t].resize(H,0.);
            }
        }

        for(size_t r = 0 ; r < R; ++r) {
            xi[r].resize(obs[r].size()-1);
            for(size_t t = 0; t < obs[r].size()-1; ++t) {
                xi[r][t].resize(H);
                for(size_t h = 0; h < H; ++h) {
                    xi[r][t][h].resize(H,0.);
                }
            }
        }
        
        for(size_t r = 0; r < R; ++r) {
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                for(size_t t = 0; t < obs[r].size(); ++t) {
                    gamma[r][t][newHidden[r][n][t]] += newProbs[r][n];
                    if(t < obs[r].size()-1) {
                        xi[r][t][newHidden[r][n][t]][newHidden[r][n][t+1]] += newProbs[r][n];
                    }
                }
            }
        }

        for(size_t r = 0; r < R; ++r) {
            for(size_t t = 0; t < obs[r].size(); ++t) {
                normalizeEps(gamma[r][t]);
                for(size_t h = 0; h < H; ++h) {
                    if(t < obs[r].size() - 1) {
                        normalizeEps(xi[r][t][h]);
                    }
                }
            }
        }

        //Translate into A,E,S
        for(size_t h = 0; h < H; ++h) {
            newS[h] *= 1. - pow(numIt, convergeFactor); 
            for(size_t g = 0; g < H; ++g) {
                newA[h][g] *= (1. - pow(numIt, convergeFactor)); 
            }
            for(size_t o = 0; o < O; ++o) {
                newE[h][o] *= (1. - pow(numIt, convergeFactor));  
            }
        }   

        for(size_t r = 0; r < R; ++r) {
            for(size_t h = 0; h < H; ++h) {
                //newS[h] += pow(numIt, 1.)*(gamma[r][0][h] - newS[h]);
                newS[h] += pow(numIt, convergeFactor)*gamma[r][0][h];
            }

            for(size_t t = 0; t < obs[r].size(); ++t) {
                for(size_t h = 0; h < H; ++h) {
                    if(t < (obs[r].size() - 1)) {
                        for(size_t g = 0; g < H; ++g) {
                            //newA[h][g] += pow(numIt, 1.)*(xi[r][t][h][g] - newA[h][g]);
                            newA[h][g] += pow(numIt, convergeFactor)*xi[r][t][h][g];
                        }
                    }

                    for(size_t o = 0; o < O; ++o) {
                        if(obs[r][t] == o) {
                            //newE[h][o] += pow(numIt, 1.)*(gamma[r][t][h] - newE[h][o]);
                            newE[h][o] += pow(numIt, convergeFactor)*gamma[r][t][h];
                        }
                    }
                }
            }
        }

        auto normalizedA = newA;
        auto normalizedE = newE;
        auto normalizedS = newS;

        normalize(normalizedS);
        for(size_t h = 0; h < H; ++h) {
            normalize(normalizedA[h]);
            normalize(normalizedE[h]);
        }

        double diff = 0.;
        for(size_t h = 0; h < H; ++h) {
            for(size_t g = 0; g < H; ++g) {
                diff = std::max(diff, std::fabs(hmm.getAEntry(h,g) - normalizedA[h][g]));
            }
        }

        hmm.setA(normalizedA);
        hmm.setE(normalizedE);
        hmm.setS(normalizedS);

        if((diff < convergence_tolerance) || (numIt > max_iteration)) {
            //Make all the 0 transitions epsilon transitions
            //If this isn't the case a bunch of stuff breaks later because we can learn infeasible models
            auto A = hmm.getA();
            auto E = hmm.getE();
            auto S = hmm.getS();

            for(auto& vec: A) {
                normalize(vec);
            }
            for(auto& vec: E) {
                normalize(vec);
            }
            normalize(S);

            hmm.setA(A);
            hmm.setS(S);
            hmm.setE(E);

            std::cout << "Algorithm took " << numIt << " iterations." << std::endl;
            break;
        }
    }
}

} //namespace chmppp
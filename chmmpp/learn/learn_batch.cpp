#include <iostream>
#include "learn.hpp"
#include "../inference/inference.hpp"
#include <iomanip>

void normalizeEps(std::vector<double> &myVec) {
    double eps = 10E-7;
    double sum = 0.;
    for(const auto &elem: myVec) sum += elem;

    if(sum != 0.) {
        for(auto &elem: myVec) elem /= sum;
    }
    else {
        for(auto &elem: myVec) elem = eps;
    }
}

void normalize(std::vector<double> &myVec) {
    double sum = 0.;
    for(const auto &elem: myVec) sum += elem;

    if(sum != 0.) {
        for(auto &elem: myVec) elem /= sum;
    }
    else {
        for(auto &elem: myVec) elem = 1./((double) myVec.size());
    }
}

namespace chmmpp {

//Each itertation of generated hidden states are underweighted by 1, 1/2, 1/3, ...
void learn_batch(HMM &hmm, const std::vector<std::vector<int>> &obs, 
                const std::vector<std::function<bool(std::vector<int>&)> >& constraintOracle,
                const std::function<std::vector<std::vector<std::vector<int>>>(
                    HMM&, const int&, const int&, const std::vector<std::vector<int>>&, 
                    const std::vector<std::function<bool(std::vector<int>&)> >&
                )> generator,
                const Options& options) 
{
    //TODO Make into options
    const double convergence_tolerance = 10E-6;
    const unsigned int max_iteration = 1000000;
    const unsigned int max_iteration_generator = 0;
    const int num_solutions = 1000; //Assumes constant number of solutions each time

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
        auto newHidden = generator(hmm, num_solutions, max_iteration_generator, obs, constraintOracle); //r,n, t
        //0 <= n < num_solutions
        /*for(size_t r = 0; r < R; ++r) {
            std::cout << std::endl;
            std::cout << "Observation:" << std::endl;
            for(size_t t = 0; t < obs[r].size(); ++t) {
                std::cout << obs[r][t];
            }
            std::cout << std::endl;
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                for(size_t t = 0; t < obs[r].size(); ++t) {
                    std::cout << newHidden[r][n][t];
                }
                std::cout << std::endl;
            }
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

        for(size_t r = 0; r < R; ++r) {
            for(size_t h = 0; h < H; ++h) {
                newS[h] += gamma[r][0][h];//*(i+1.)/((double)numIt);
            }

            for(size_t t = 0; t < obs[r].size(); ++t) {
                for(size_t h = 0; h < H; ++h) {
                    if(t < (obs[r].size() - 1)) {
                        for(size_t g = 0; g < H; ++g) {
                            newA[h][g] += xi[r][t][h][g];//*(i+1.)/((double)numIt);
                        }
                    }

                    for(size_t o = 0; o < O; ++o) {
                        if(obs[r][t] == o) {
                            newE[h][o] += gamma[r][t][h];//*(i+1.)/((double)numIt);
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
                diff = std::max(diff, std::abs(hmm.getAEntry(h,g) - normalizedA[h][g]));
            }
        }

        hmm.setA(normalizedA);
        hmm.setE(normalizedE);
        hmm.setS(normalizedS);

        if((diff < convergence_tolerance) || (numIt > max_iteration)) {
            break;
        }

        std::cout << std::setprecision(6) << numIt << " " << diff << std::endl;
    }

}

} //namespace chmppp
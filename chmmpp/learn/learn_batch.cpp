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
    double eps = 1E-100;
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
    const double convergence_tolerance = 1.E-6;
    const unsigned int max_iteration = 1000000;
    const unsigned int max_iteration_generator = 0;
    //TODO this doesn't work right now const int num_solutions = 100; //Assumes constant number of solutions each time 
    const double convergeFactor = -1.; //Should be in [-1,-0.5)

    std::vector<std::vector<std::vector<std::vector<double>>>> gamma;
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> xi; 

    const int H = hmm.getH();
    const int O = hmm.getO();
    const int R = obs.size();

    int numIt = 0;

    std::vector<std::vector<double>> AStar(H);
    std::vector<std::vector<double>> EStar(H);
    std::vector<double> SStar(H,0.);
    for(auto &vec: AStar) vec.resize(H,0.);
    for(auto &vec: EStar) vec.resize(O,0.);

    while(true) {
        ++numIt;//==gamma.size();
        auto newHidden = generator(hmm, obs); //r,n,t

        /*for(size_t r = 0; r < newHidden.size(); ++r) {
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                for(size_t t = 0; t < newHidden[r][n].size(); ++t) {
                    std::cout << newHidden[r][n][t];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }*/

        const double xi = pow(numIt, convergeFactor);

        std::vector<std::vector<long double>> p(obs.size()); //r,n
        for(size_t r = 0; r < obs.size(); ++r) { 
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                p[r].push_back(1);
            }
        }

        for(size_t h = 0; h < H; ++h) {
            SStar[h] *= (1.-xi);
            for(size_t h2 = 0; h2 < H; ++h2) AStar[h][h2] *= (1-xi);
            for(size_t o = 0; o < O; ++o) EStar[h][o] *= (1-xi);
        }

        for(size_t r = 0; r < newHidden.size(); ++r) {
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                SStar[newHidden[r][n][0]] += p[r][n]*xi;
            }
        }

        for(size_t r = 0; r < newHidden.size(); ++r) {
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                for(size_t t = 0; t < newHidden[r][n].size()-1; ++t) {
                    AStar[newHidden[r][n][t]][newHidden[r][n][t+1]] += p[r][n]*xi;
                }
            }
        }

        for(size_t r = 0; r < newHidden.size(); ++r) {
            for(size_t n = 0; n < newHidden[r].size(); ++n) {
                for(size_t t = 0; t < newHidden[r][n].size(); ++t) {
                    ++EStar[newHidden[r][n][t]][obs[r][t]] += p[r][n]*xi;
                }
            }
        }

        auto normalizedA = AStar;
        auto normalizedE = EStar;
        auto normalizedS = SStar;

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

        //std::cout << std::scientific << numIt << " " << diff << std::endl;

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
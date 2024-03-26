// HMM.cpp

#include <iostream>
#include "HMM.hpp"

namespace chmmpp {

/*
TODO
----

- Simplify Baum-Welch Style Algorithms with repeated functions
- Fix Monte Carlo method to not break when we have 0's in the transition matrix

*/

// Return a random number from 0,1
// Needs to be an internal function b/c we are calling random a bunch of different times in the run
// function, possibly multiple times

double HMM::getRandom() { return dist(generator); }

HMM::HMM(long int seed) { initialize(A, S, E, seed); }

HMM::HMM(const std::vector<std::vector<double> > &inputA, const std::vector<double> &inputS,
         const std::vector<std::vector<double> > &inputE, int long seed)
{
    initialize(inputA, inputS, inputE, seed);
}

void HMM::initialize(const std::vector<std::vector<double> > &inputA,
                     const std::vector<double> &inputS,
                     const std::vector<std::vector<double> > &inputE, long int seed)
{
    H = inputA.size();

    if (H > 0) {
        // Check if sizes are correct
        if ((inputS.size() != H) || (inputE.size() != H)) {
            std::cout << "Error in constructor for HMM, matrices not appropriately sized."
                      << std::endl;
            throw std::exception();
        }

        for (size_t h = 0; h < H; ++h) {
            if (inputA[h].size() != H) {
                std::cout << "Error in constructor for HMM, A is not a square matrix." << std::endl;
                throw std::exception();
            }
        }

        O = inputE[0].size();

        for (size_t h = 0; h < H; ++h) {
            if (inputE[h].size() != O) {
                std::cout << "Error in constructor for HMM, E is not a matrix." << std::endl;
                throw std::exception();
            }
        }

        // Check if matrices represent probabilities
        double sum = 0;
        for (size_t h1 = 0; h1 < H; ++h1) {
            sum = 0;
            for (size_t h2 = 0; h2 < H; ++h2) {
                if (inputA[h1][h2] < 0.) {
                    std::cout << "Error in constructor for HMM, A cannot have negative entries."
                              << std::endl;
                    throw std::exception();
                }

                sum += inputA[h1][h2];
            }

            if (std::abs(sum - 1.) > 10E-6) {
                std::cout << "Error in constructor for HMM, the rows of A must sum to 1."
                          << std::endl;
                throw std::exception();
            }
        }

        sum = 0;
        for (size_t h = 0; h < H; ++h) {
            if (inputS[h] < 0.) {
                std::cout << "Error in constructor for HMM, S cannot have negative entries."
                          << std::endl;
                throw std::exception();
            }
            sum += inputS[h];
        }
        if (std::abs(sum - 1.) > 10E-6) {
            std::cout << "Error in constructor for HMM, the entries of S must sum to 1."
                      << std::endl;
        }

        for (size_t h = 0; h < H; ++h) {
            sum = 0;
            for (size_t o = 0; o < O; ++o) {
                if (inputE[h][o] < 0.) {
                    std::cout << "Error in constructor for HMM, E cannot have negative entries."
                              << std::endl;
                    throw std::exception();
                }

                sum += inputE[h][o];
            }

            if (std::abs(sum - 1.) > 10E-6) {
                std::cout << "Error in constructor for HMM, the rows of E must sum to 1."
                          << std::endl;
                throw std::exception();
            }
        }
    }

    A = inputA;
    S = inputS;
    E = inputE;

    std::random_device rand_dev;
    std::mt19937 myGenerator(rand_dev());
    generator = myGenerator;
    generator.seed(seed);
    std::uniform_real_distribution<double> myDist(0., 1.);
    dist = myDist;
}

//----------------------------------
//-----Access private variables-----
//----------------------------------

size_t HMM::getH() const { return H; }

size_t HMM::getO() const { return O; }

std::vector<std::vector<double> > HMM::getA() const { return A; }

std::vector<double> HMM::getS() const { return S; }

std::vector<std::vector<double> > HMM::getE() const { return E; }

// Range not checked for speed
double HMM::getAEntry(size_t h1, size_t h2) const { return A[h1][h2]; }

double HMM::getSEntry(size_t h) const { return S[h]; }

double HMM::getEEntry(size_t h, size_t o) const { return E[h][o]; }

//-----------------------
//-----Print the HMM-----
//-----------------------

void HMM::printS() const
{
    std::cout << "Start vector:\n";
    for (size_t i = 0; i < H; ++i) {
        std::cout << S[i] << " ";
    }
    std::cout << "\n\n";
    return;
}

void HMM::printA() const
{
    std::cout << "Transmission matrix:\n";
    for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j < H; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    return;
}

void HMM::printO() const
{
    std::cout << "Emission matrix: (Columns are hidden states, rows are observed states)\n";
    for (size_t h = 0; h < H; ++h) {
        for (size_t o = 0; o < O; ++o) {
            std::cout << E[h][o] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";
    return;
}

void HMM::print() const
{
    printS();
    printA();
    printO();
    return;
}

//---------------------
//-----Run the HMM-----
//---------------------

// This generates the observed states and hidden states running the HMM for T time steps
// Not const b/c of the random stuff
void HMM::run(int T, std::vector<int> &observedStates, std::vector<int> &hiddenStates)
{
    observedStates.clear();
    hiddenStates.clear();

    // Initial Hidden State
    double startProb = getRandom();
    double prob = 0;
    for (size_t h = 0; h < H; ++h) {
        prob += S[h];
        if (startProb < prob) {
            hiddenStates.push_back(h);
            break;
        }
    }

    // Initial Observed State
    double obsProb = getRandom();
    prob = 0;
    for (size_t o = 0; o < O; ++o) {
        prob += E[hiddenStates[0]][o];
        if (obsProb < prob) {
            observedStates.push_back(o);
            break;
        }
    }

    // All other states
    for (int t = 1; t < T; ++t) {
        startProb = getRandom();
        prob = 0;
        for (size_t h = 0; h < H; ++h) {
            prob += A[hiddenStates[t - 1]][h];
            if (startProb < prob) {
                hiddenStates.push_back(h);
                break;
            }
        }

        obsProb = getRandom();
        prob = 0;
        for (size_t o = 0; o < O; ++o) {
            prob += E[hiddenStates[t]][o];
            if (obsProb < prob) {
                observedStates.push_back(o);
                break;
            }
        }
    }

    return;
}

}  // namespace chmmpp

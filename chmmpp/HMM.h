#pragma once

#include <functional>
#include <vector>
#include <random>

namespace chmmpp {

/*
TODO
----

- Simplify Baum-Welch Style Algorithms with repeated functions
- Fix Monte Carlo method to not break when we have 0's in the transition matrix

*/

// This is the class for dealing with HMMs
// It stores (as protected variables) the transition matrix (A), start probabilities (S), and
// Emission Probabilities (E) as well as the number of hidden states (H) and observed states (O)
class HMM {
   protected:
    size_t H;                                     // Number of hidden states
    size_t O;                                     // Number of observed states
    std::vector<std::vector<double> > A;          // Transition matrix, size HxH
    std::vector<double> S;                        // Start probs, size H
    std::vector<std::vector<double> > E;          // Emission probs, size HxO
    std::mt19937 generator;                       // Needed for running the HMM
    std::uniform_real_distribution<double> dist;  // Ditto

    double getRandom();

   public:
    HMM(long int seed = time(NULL));

    HMM(const std::vector<std::vector<double> > &inputA, const std::vector<double> &inputS,
        const std::vector<std::vector<double> > &inputE, long int seed = time(NULL));

    void initialize(const std::vector<std::vector<double> > &inputA,
                    const std::vector<double> &inputS,
                    const std::vector<std::vector<double> > &inputE, long int seed);

    // Get Private Variables
    size_t getH() const;
    size_t getO() const;
    std::vector<std::vector<double> > getA() const;
    std::vector<double> getS() const;
    std::vector<std::vector<double> > getE() const;
    double getAEntry(size_t h1, size_t h2) const;
    double getSEntry(size_t h) const;
    double getEEntry(size_t h, size_t o) const;

    void printS() const;
    void printA() const;
    void printO() const;
    void print() const;

    void run(int T, std::vector<int> &observedStates, std::vector<int> &hiddenStates);

    // Inference
    std::vector<int> aStar(const std::vector<int> &observations, double &logProb) const;
    std::vector<int> aStar(const std::vector<int> &observations, double &logProb,
                           const int numZeros) const;
    std::vector<int> aStarOracle(
        const std::vector<int> &observations, double &logProb,
        const std::function<bool(std::vector<int>)> &constraintOracle) const;
    std::vector<std::vector<int> > aStarMult(const std::vector<int> &observations, double &logProb,
                                             const int numZeros, const int numSolns) const;
    std::vector<std::vector<int> > aStarMult(
        const std::vector<int> &observations, double &logProb,
        const std::function<bool(std::vector<int>)> &constraintOracle, const int numSolns) const;

    double logProb(const std::vector<int> obs, const std::vector<int> guess) const;

    // Learning Algorithms
    void learn(const std::vector<int> &obs, const int numZeros, const double eps = 10E-6);
    void learn(const std::vector<std::vector<int> > &obs, const std::vector<int> &numZeros,
               const double eps = 10E-6);
    void learn(const std::vector<int> &obs, const double eps = 10E-6);
    void learn(const std::vector<std::vector<int> > &obs, const double eps = 10E-6);
    void learn(const std::vector<std::vector<int> > &obs,
               const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
               const double eps = 10E-6, const int C = 10E4);
    void learnHard(const std::vector<std::vector<int> > &obs,
                   const std::vector<std::function<bool(std::vector<int>)> > &constraintOracle,
                   const double eps = 10E-6, int numSolns = 1);
};

}  // namespace chmmpp

#pragma once

#include <vector>
#include <random>
#include <chmmpp/util/Options.hpp>

namespace chmmpp {

// This is the class for dealing with HMMs
// It stores (as protected variables) the transition matrix (A), start probabilities (S), and
// Emission Probabilities (E) as well as the number of hidden states (H) and observed states (O)
class HMM : public Options {
   protected:
    size_t H;                                     // Number of hidden states
    size_t O;                                     // Number of observed states
    std::vector<std::vector<double> > A;          // Transition matrix, size HxH
    std::vector<double> S;                        // Start probs, size H
    std::vector<std::vector<double> > E;          // Emission probs, size HxO
    std::mt19937 generator;                       // Needed for running the HMM
    std::uniform_real_distribution<double> dist;  // Ditto

    long int seed;
    double getRandom();

   public:
    HMM(long int seed = time(NULL));

    HMM(const std::vector<std::vector<double> > &inputA, const std::vector<double> &inputS,
        const std::vector<std::vector<double> > &inputE, long int seed = time(NULL));

    void initialize(const std::vector<std::vector<double> > &inputA,
                    const std::vector<double> &inputS,
                    const std::vector<std::vector<double> > &inputE);
    void set_seed(long int seed);
    void reset_rng();

    void initialize_from_file(const std::string &json_filename);

    void initialize_from_string(const std::string &json_string);

    // Get Private Variables
    size_t getH() const;
    size_t getO() const;
    std::vector<std::vector<double> > getA() const;
    std::vector<double> getS() const;
    std::vector<std::vector<double> > getE() const;
    double getAEntry(size_t h1, size_t h2) const;
    double getSEntry(size_t h) const;
    double getEEntry(size_t h, size_t o) const;

    void setA(std::vector<std::vector<double> > newA);
    void setS(std::vector<double> newS);
    void setE(std::vector<std::vector<double> > newE);

    void printS() const;
    void printA() const;
    void printO() const;
    void print() const;

    void run(int T, std::vector<int> &observedStates, std::vector<int> &hiddenStates);

    double logProb(const std::vector<int> &obs, const std::vector<int> &hidden_states) const;

    //
    // Inference with the Viterbi algorithm
    //
    void viterbi(const std::vector<int> &observations, std::vector<int> &hidden_states,
                 double &logprob);

    //
    // Inference with the A* algorithm
    //
    void aStar(const std::vector<int> &observations, std::vector<int> &hidden_states,
               double &logprob);
    //
    // Inference with a linear programming solver
    //
    // Options
    //      solver_name (string):   Name of the linear programming solver (Default: gurobi).
    //      keep_data (int):        If 1, then keep data used to generate the LP model (Default: 0).
    //      y_binary (int):         If 1, then use binary flow variables in the LP model (Default:
    //      0). debug (int):            If 1, then print debugging information (Default: 0).
    //
    void lp_map_inference(const std::vector<int> &observations, std::vector<int> &hidden_states,
                          double &logprob);

    // Estimate the HMM parameters using the values of hidden states
    void estimate_hmm(const std::vector<int> &obs, const std::vector<int> &hid);
    void estimate_hmm(const std::vector<std::vector<int> > &obs,
                      const std::vector<std::vector<int> > &hid);

    //
    // Baum-Welch learning algorithm
    //
    //  Options
    //      convergence_tolerance (double):     Stop learning if solution tolerance is below this
    //      threshold
    //                                          (Default: 10E-6).
    //      max_iterations: unsigned int        Stop learning if number of iterations equals this
    //      threshold.
    //                                          No threshold if this is 0 (Default: 0).
    //
    void baum_welch(const std::vector<int> &obs);
    void baum_welch(const std::vector<std::vector<int> > &obs);
};

}  // namespace chmmpp

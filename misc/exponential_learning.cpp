#include <iostream>
#include <chmmpp/chmmpp.hpp>

int main() {
    int r = 200;
    auto eps = (1./4.);

    std::vector<std::vector<double>> A{{0.5, 0.5}, {0.5, 0.5}};  // Transition Matrix
    std::vector<double> S = {1, 0};                          // Start probabilities
    std::vector<std::vector<double>> E{{1-eps, eps}, {eps, 1-eps}};  // Emission Matrix

    // Create HMM
    chmmpp::HMM hmm(A, S, E, 0);
    hmm.print();

    std::vector<std::vector<int>> obs;
    obs.resize(2*r);

    for(size_t j = 0; j < r; ++j) {
        obs[j].resize(2*r);
    for(size_t i = 0; i < r; ++i) {
        obs[j][i] = 0;
        obs[j][i+r] = 1;
    }
    }
    
    for(size_t j = r; j < 2*r; ++j) {
        obs[j].resize(2*r); 
    for(size_t i = 0; i < 2*r; ++i) {
        if(!(i % 2)) obs[j][i] = 0;
        else obs[j][i] = 1;
    }
    }

    hmm.baum_welch(obs);
    hmm.print();
}
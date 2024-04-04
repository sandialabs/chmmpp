#include <iostream>
#include "learn.hpp"

namespace chmmpp {

void estimate_hmm(HMM& hmm, const std::vector<std::vector<int>>& obs,
                  const std::vector<std::vector<int>>& hid)
{
    auto A = hmm.getA();
    auto S = hmm.getS();
    auto E = hmm.getE();
    auto H = hmm.getH();
    auto O = hmm.getO();

    size_t T = obs[0].size();
    size_t R = obs.size();

    std::vector<size_t> Scount(H);
    for (size_t i = 0; i < R; i++) Scount[hid[i][0]]++;
    for (size_t j = 0; j < H; j++) S[j] = Scount[j] / ((double)R);

    std::vector<std::vector<size_t>> Acount(H);
    for (auto& a : Acount) a.resize(H);
    for (size_t i = 0; i < R; i++)
        for (size_t t = 0; t < T - 1; ++t) Acount[hid[i][t]][hid[i][t + 1]]++;
    for (size_t a = 0; a < H; a++) {
        double total = 0;
        for (size_t b = 0; b < H; b++) {
            total += Acount[a][b];
            // std::cout << "Acount " << a << " " << b << " " << Acount[a][b] << std::endl;
        }
        if(total != 0) {
            for (size_t b = 0; b < H; b++) {
                A[a][b] = Acount[a][b] / total;
                // std::cout << "A " << a << " " << b << " " << A[a][b] << std::endl;
            }
        }
        else {
            for(size_t b = 0; b < H; ++b) {
                A[a][b] = 1./H;
            }
        }
    }

    std::vector<std::vector<size_t>> Ecount(H);
    for (auto& e : Ecount) e.resize(O);
    for (size_t i = 0; i < R; i++)
        for (size_t t = 0; t < T; ++t) Ecount[hid[i][t]][obs[i][t]]++;
    for (size_t a = 0; a < H; a++) {
        double total = 0;
        for (size_t b = 0; b < O; b++) {
            total += Ecount[a][b];
            // std::cout << "Ecount " << a << " " << b << " " << Ecount[a][b] << std::endl;
        }
        if(total != 0) {
            for (size_t b = 0; b < O; b++) {
                E[a][b] = Ecount[a][b] / total;
                // std::cout << "E " << a << " " << b << " " << E[a][b] << std::endl;
            }
        }
        else {
            for(size_t b = 0; b < O; ++b) {
                E[a][b] = 1./O;
            }
        }
    }

    hmm.initialize(A, S, E);
}

}  // namespace chmmpp

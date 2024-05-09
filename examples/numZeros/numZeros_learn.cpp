#include <iostream>
#include "numZerosHMM.hpp"

namespace chmmpp {

namespace {

// A customized Soft EM???
void local_learn_numZeros(HMM &hmm, const std::vector<std::vector<int>> &obs,
                          const std::vector<int> &numZeros, const double convergence_tolerance)
{
    auto A = hmm.getA();
    auto S = hmm.getS();
    auto E = hmm.getE();
    auto H = hmm.getH();
    auto O = hmm.getO();
    int T = obs[0].size();
    size_t R = obs.size();

    while (true) {
        std::vector<std::vector<std::vector<double>>> totalGamma;
        std::vector<std::vector<std::vector<std::vector<double>>>> totalXi;
        for (size_t r = 0; r < R; ++r) {
            // alpha
            std::vector<std::vector<std::vector<double>>>
                alpha;  // alpha[c][h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta, c
                        // 0's)
            alpha.resize(numZeros[r] + 1);
            for (int c = 0; c <= numZeros[r]; ++c) {
                alpha[c].resize(H);
                for (size_t h = 0; h < H; ++h) {
                    alpha[c][h].resize(T);

                    if (((c == 1) && (h == 0)) || ((c == 0) && (h != 0))) {
                        alpha[c][h][0] = S[h] * E[h][obs[r][0]];
                    }

                    else {
                        alpha[c][h][0] = 0.;
                    }
                }
            }

            for (int t = 1; t < T - 1; ++t) {
                for (int c = 0; c <= numZeros[r]; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        alpha[c][h][t] = 0.;
                        for (size_t h1 = 0; h1 < H; ++h1) {
                            int oldC = c;
                            if (h1 == 0) {
                                --oldC;
                            }

                            if (oldC >= 0) {
                                alpha[c][h][t] += alpha[oldC][h1][t - 1] * A[h1][h];
                            }
                        }
                        alpha[c][h][t] *= E[h][obs[r][t]];
                    }
                }
            }

            // t = T-1
            for (int c = 0; c <= numZeros[r]; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    alpha[c][h][T - 1] = 0.;
                    if (c == numZeros[r]) {
                        for (size_t h1 = 0; h1 < H; ++h1) {
                            int oldC = c;
                            if (h1 == 0) {
                                --oldC;
                            }

                            if (oldC >= 0) {
                                alpha[c][h][T - 1] += alpha[oldC][h1][T - 2] * A[h1][h];
                            }
                        }
                        alpha[c][h][T - 1] *= E[h][obs[r][T - 1]];
                    }
                }
            }

            // beta
            std::vector<std::vector<std::vector<double>>>
                beta;  // beta[c][h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta,
                       // c 0's )
            beta.resize(numZeros[r] + 1);
            for (int c = 0; c <= numZeros[r]; ++c) {
                beta[c].resize(H);
                for (size_t h = 0; h < H; ++h) {
                    beta[c][h].resize(T);

                    if (c == 0) {
                        beta[c][h][T - 1] = 1;
                    }

                    else {
                        beta[c][h][T - 1] = 0;
                    }
                }
            }

            for (int t = T - 2; t > 0; --t) {
                for (int c = 0; c <= numZeros[r]; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        beta[c][h][t] = 0.;
                        for (size_t h2 = 0; h2 < H; ++h2) {
                            int newC = c;
                            if (h2 == 0) {
                                --newC;
                            }

                            if (newC >= 0) {
                                beta[c][h][t]
                                    += beta[newC][h2][t + 1] * A[h][h2] * E[h2][obs[r][t + 1]];
                            }
                        }
                    }
                }
            }

            // t = 0
            for (int c = 0; c <= numZeros[r]; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    beta[c][h][0] = 0.;
                }
            }

            // h[0] = 0
            if (numZeros[r] > 0) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    int newC = numZeros[r] - 1;
                    if (h2 == 0) {
                        --newC;
                    }
                    if (newC >= 0) {
                        beta[numZeros[r] - 1][0][0]
                            += beta[newC][h2][1] * A[0][h2] * E[h2][obs[r][1]];
                    }
                }
            }

            // h[0] != 0
            for (size_t h = 1; h < H; ++h) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    int newC = numZeros[r];
                    if (h2 == 0) {
                        --newC;
                    }

                    if (newC >= 0) {
                        beta[numZeros[r]][h][0] += beta[newC][h2][1] * A[h][h2] * E[h2][obs[r][1]];
                    }
                }
            }

            // den = P(O | theta)
            // Need different denominators because of the scaling
            // This is numerically a VERY weird algorithm
            std::vector<double> den;
            for (int t = 0; t < T; ++t) {
                den.push_back(0.);
                for (size_t h = 0; h < H; ++h) {
                    for (int c = 0; c <= numZeros[r]; ++c) {
                        den[t] += alpha[c][h][t] * beta[numZeros[r] - c][h][t];
                    }
                }
            }

            // Gamma
            std::vector<std::vector<double>> gamma;  // gamma[h][t] = P(H_t = h | Y , theta)
            gamma.resize(H);
            for (size_t h = 0; h < H; ++h) {
                gamma[h].resize(T);
            }

            for (size_t h = 0; h < H; ++h) {
                for (int t = 0; t < T; ++t) {
                    double num = 0.;
                    for (int c = 0; c <= numZeros[r]; ++c) {
                        num += alpha[c][h][t] * beta[numZeros[r] - c][h][t];
                    }
                    gamma[h][t] = num / den[t];
                }
            }

            totalGamma.push_back(gamma);

            // xi
            std::vector<std::vector<std::vector<double>>>
                xi;  // xi[i][j][t] = P(H_t = i, H_t+1 = j, O| theta)
            xi.resize(H);
            for (size_t h1 = 0; h1 < H; ++h1) {
                xi[h1].resize(H);
                for (size_t h2 = 0; h2 < H; ++h2) {
                    xi[h1][h2].resize(T - 1);
                }
            }

            for (size_t h1 = 0; h1 < H; ++h1) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    for (int t = 0; t < T - 1; ++t) {
                        double num = 0.;

                        for (int c = 0; c <= numZeros[r]; ++c) {
                            int middleC = 0;
                            if (h2 == 0) {
                                ++middleC;
                            }

                            if (numZeros[r] - middleC - c >= 0) {
                                num += alpha[c][h1][t] * beta[numZeros[r] - middleC - c][h2][t + 1];
                            }
                        }
                        num *= A[h1][h2] * E[h2][obs[r][t + 1]];

                        xi[h1][h2][t] = num / den[t];
                    }
                }
            }

            totalXi.push_back(xi);
        }

        // New S
        for (size_t h = 0; h < H; ++h) {
            S[h] = 0.;
            for (size_t r = 0; r < R; ++r) {
                S[h] += totalGamma[r][h][0];
            }
            S[h] /= R;
        }
        hmm.setS(S);

        // New E
        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t o = 0; o < O; ++o) {
                    double num = 0.;
                    double newDen = 0.;

                    for (int t = 0; t < T; ++t) {
                        if (obs[r][t] == o) {
                            num += totalGamma[r][h][t];
                        }
                        newDen += totalGamma[r][h][t];
                    }

                    E[h][o] = num / newDen;
                }
            }
        }
        hmm.setE(E);

        double tol = 0.;

        // New A
        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double num = 0.;
                double newDen = 0.;
                for (size_t r = 0; r < R; ++r) {
                    for (int t = 0; t < T - 1; ++t) {
                        num += totalXi[r][h1][h2][t];
                        newDen += totalGamma[r][h1][t];
                    }
                }
                tol = std::max(std::abs(A[h1][h2] - num / newDen), tol);
                A[h1][h2] = num / newDen;
            }
        }
        hmm.setA(A);

        // std::cout << "Tolerance: " << tol << "\n";
        //  tol = 0.;
        if (tol < convergence_tolerance) {
            break;
        }
    }
}

}  // namespace

void numZerosHMM::learn_numZeros(const std::vector<std::vector<int>> &obs)
{
    std::vector<int> newNumZeros;
    for (size_t i = 0; i < obs.size(); i++) newNumZeros.push_back(numZeros);

    auto option = get_option<double>("convergence_tolerance");
    double convergence_tolerance = 10E-6;
    if (option.has_value()) convergence_tolerance = *option;
    local_learn_numZeros(hmm, obs, newNumZeros, convergence_tolerance);
}

void numZerosHMM::learn_numZeros(const std::vector<int> &obs)
{
    std::vector<std::vector<int>> newObs;
    newObs.push_back(obs);
    learn_numZeros(newObs);
}

}  // namespace chmmpp

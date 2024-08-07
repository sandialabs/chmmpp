#include <iostream>
#include "learn.hpp"

namespace chmmpp {

namespace {

void process_options(Options& options, double& convergence_tolerance, unsigned int& max_iterations, unsigned int& quiet)
{
    for (auto& it : options.option_data) {
        if (it.first == "convergence_tolerance") {
            if (std::holds_alternative<double>(it.second))
                convergence_tolerance = std::get<double>(it.second);
        }
        else if (it.first == "quiet") {
            if (std::holds_alternative<unsigned int>(it.second))
                quiet = std::get<unsigned int>(it.second);
        }
        else if (it.first == "max_iterations") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp > 0)
                    max_iterations = tmp;
                else
                    std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer"
                              << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second))
                max_iterations = std::get<unsigned int>(it.second);
            else
                std::cerr << "WARNING: 'max_iterations' option must be a non-negative integer"
                          << std::endl;
        }
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

}  // namespace

//
// TODO - Move normalization operations into the HMM class.
//
// WEH - This is only done at the end.  Hence, it would make sense to have this operation be done
//          within the HMM class when setting the data there. If this is only being done to resolve
//          issues within the MIP, then maybe we should only do the normalization before the MIP
//          operations???
//

//
// TODO - Simplify memory operations
//
// WEH - This code does lots of unnecessary memory operations, which will really slow it down.
//          The HMM data is repeated set into the HMM.  Then, there is a final get/set to
//          normalize the matrices.  But I think the code could be simplified to only set
//          the HMM data at the end of this function.
//       Similarly, the totalGamma and totalXi data objects are iteratively constructed with push_back
//          operations that are redundant from iteration to iteration.  I think we can setup this memory
//          once and then over-write it in each iteration.
//
void learn_unconstrained(HMM& hmm, const std::vector<std::vector<int> >& obs)
{
    double convergence_tolerance = 10E-6;
    unsigned int max_iterations = 10000000;
    unsigned int quiet=0;
    process_options(hmm.get_options(), convergence_tolerance, max_iterations, quiet);
    
    auto A = hmm.getA();
    auto S = hmm.getS();
    auto E = hmm.getE();
    auto H = hmm.getH();
    auto O = hmm.getO();

    size_t R = obs.size();
    size_t numIt = 0;

    if (not quiet)
        std::cout << "learn_unconstrained" << std::endl;
    while (true) {
        ++numIt;
        std::vector<std::vector<std::vector<double> > > totalGamma;
        std::vector<std::vector<std::vector<std::vector<double> > > > totalXi;

        for (size_t r = 0; r < R; ++r) {
            size_t T = obs[r].size();
            // alpha
            std::vector<std::vector<double> >
                alpha;  // alpha[h][t] = P(O_0 = obs[0], ... ,O_t = obs[t], H_t = h | theta)
            alpha.resize(H);
            for (size_t h = 0; h < H; ++h) {
                alpha[h].resize(T, 0.);
                alpha[h][0] = S[h] * E[h][obs[r][0]];
            }

            for (int t = 1; t < T; ++t) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t h1 = 0; h1 < H; ++h1) {
                        alpha[h][t] += alpha[h1][t - 1] * A[h1][h];
                    }

                    alpha[h][t] *= E[h][obs[r][t]];
                }
            }

            //Normalize alpha to improve numerical underflow
            //Have to be VERY careful with normalizing after doing this
            for(size_t t = 0; t < T; ++t) {
                double sum = 0.;
                for(size_t h = 0; h < H; ++h) {
                    sum += alpha[h][t];
                }
                for(size_t h = 0; h < H; ++h) {
                    if(sum != 0) alpha[h][t] /= sum;
                    else alpha[h][t] = 0.; //Works b/c alpha >=0. 
                }
            }

            // beta
            std::vector<std::vector<double> >
                beta;  // beta[h][t] = P(O_{t+1} = o_{t+1} ... O_{T-1} = o_{T-1} | H_t = h theta)
            beta.resize(H);
            for (size_t h = 0; h < H; ++h) {
                beta[h].resize(T);
                beta[h][T - 1] = 1.;
            }

            for (int t = T - 2; t >= 0; --t) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t h2 = 0; h2 < H; ++h2) {
                        beta[h][t] += beta[h2][t + 1] * A[h][h2] * E[h2][obs[r][t + 1]];
                    }
                }
            }

            //Normalize beta to improve numerical underflow
            //CAREFUL with normalization!
            for(size_t t = 0; t < T; ++t) {
                double sum = 0.;
                for(size_t h = 0; h < H; ++h) {
                    sum += beta[h][t];
                }
                for(size_t h = 0; h < H; ++h) {
                    if(sum != 0) beta[h][t] /= sum;
                    else beta[h][t] = 0.; //Works b/c beta >=0. 
                }
            }

            // den = P(O | theta)
            std::vector<double> gamma_den(T);
            for(size_t t = 0; t < T; ++t) { 
                for (size_t h = 0; h < H; ++h) {
                    gamma_den[t] += alpha[h][t] * beta[h][t];
                }
            }

            // Gamma
            std::vector<std::vector<double> > gamma;  // gamma[h][t] = P(H_t = h | Y , theta)
            gamma.resize(H);
            for (size_t h = 0; h < H; ++h) {
                gamma[h].resize(T);
            }

            for (size_t h = 0; h < H; ++h) {
                for (int t = 0; t < T; ++t) {
                    gamma[h][t] = alpha[h][t] * beta[h][t] / gamma_den[t];
                }
            }
            totalGamma.push_back(gamma);

            // xi
            std::vector<double> xi_den(T-1);
            for(size_t t = 0; t < T-1; ++t) {
                for(size_t h1 = 0; h1 < H; ++h1) {
                    for(size_t h2 = 0; h2 < H; ++h2) {
                       xi_den[t] += alpha[h1][t] * beta[h2][t + 1] * A[h1][h2]
                                        * E[h2][obs[r][t + 1]];
                    }
                }
            }

            std::vector<std::vector<std::vector<double> > >
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
                        xi[h1][h2][t] = alpha[h1][t] * beta[h2][t + 1] * A[h1][h2]
                                        * E[h2][obs[r][t + 1]] / xi_den[t];
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
        for (size_t h = 0; h < H; ++h) {
            for (size_t o = 0; o < O; ++o) {
                double num = 0.;
                double newDen = 0.;
                for (size_t r = 0; r < R; ++r) {
                    for (int t = 0; t < obs[r].size(); ++t) {
                        if (obs[r][t] == o) {
                            num += totalGamma[r][h][t];
                        }
                        newDen += totalGamma[r][h][t];
                    }
                }
                E[h][o] = num / newDen;
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
                    for (int t = 0; t < obs[r].size() - 1; ++t) {
                        num += totalXi[r][h1][h2][t];
                        newDen += totalGamma[r][h1][t];
                    }
                }
                tol = std::max(std::fabs(A[h1][h2] - num / newDen), tol);
                A[h1][h2] = num / newDen;
            }
        }
        hmm.setA(A);

        if (not quiet) {
            //std::cout << "  Tolerance: " << tol << std::endl;
            //std::cout << "  Iteration: " << numIt << std::endl;
        }

        if (tol < convergence_tolerance) {
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
            
            if (not quiet)
                std::cout << "Terminated due to convergence tolerance: " << tol << " < " << convergence_tolerance << std::endl;
            break;
        }
        if (max_iterations and (numIt >= max_iterations)) {
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

            if (not quiet)
                std::cout << "Terminated due to iteration limit: " << numIt << std::endl;
            break;
        }
    }
}

void learn_unconstrained(HMM& hmm, const std::vector<int>& obs)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    learn_unconstrained(hmm, newObs);
}

}  // namespace chmmpp

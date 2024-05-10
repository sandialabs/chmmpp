#include <iostream>
#include "chmmpp/learn/LearnStochastic.hpp"

namespace chmmpp {

namespace {

void process_options(const Options &options, double &convergence_tolerance, unsigned int &C)
{
    for (const auto &it : options.options) {
        if (it.first == "C") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp > 0)
                    C = static_cast<unsigned int>(tmp);
                else
                    std::cerr << "WARNING: 'C' option must be a non-negative integer" << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second)) {
                C = std::get<unsigned int>(it.second);
            }
            else
                std::cerr << "WARNING: 'C' option must be a non-negative integer" << std::endl;
        }
        else if (it.first == "convergence_tolerance") {
            if (std::holds_alternative<double>(it.second))
                convergence_tolerance = std::get<double>(it.second);
            else
                std::cerr << "WARNING: 'convergence_tolerance' option must be a double"
                          << std::endl;
        }
    }
}

}  // namespace

// Will work best/fastest if the sets of hidden states which satisfy the constraints
// This algorithm is TERRIBLE, I can't even get it to converge in a simple case with T = 10.
// This is currently the only learning algorithm we have for having a constraint oracle rather than
// ``simple'' constraints This also fails to work if we are converging towards values in the
// transition matrix with 0's (which is NOT uncommon)
void LearnStochastic::learn(const std::vector<std::vector<int> > &obs,
                            const double convergence_tolerance, const int C)
{
    auto A = hmm.getA();
    auto S = hmm.getS();
    auto E = hmm.getE();
    auto H = hmm.getH();
    auto O = hmm.getO();

    size_t R = obs.size();
    int totTime = 0.;
    for (size_t r = 0; r < R; ++r) {
        totTime += obs[r].size();
    }

    std::vector<std::vector<double> > SStar(R);
    std::vector<std::vector<std::vector<double> > > AStar(R);
    std::vector<std::vector<std::vector<double> > > EStar(R);

    std::vector<std::vector<int> > SStarCounter(R);
    std::vector<std::vector<std::vector<int> > > AStarCounter(R);
    std::vector<std::vector<std::vector<int> > > EStarCounter(R);

    for (size_t r = 0; r < R; ++r) {
        SStar[r].resize(H);
        AStar[r].resize(H);
        EStar[r].resize(H);
        SStarCounter[r].resize(R);
        AStarCounter[r].resize(R);
        EStarCounter[r].resize(R);

        for (size_t h = 0; h < H; ++h) {
            AStar[r][h].resize(H);
            AStarCounter[r][h].resize(H);
            EStar[r][h].resize(O);
            EStarCounter[r][h].resize(O);
        }
    }

    std::vector<std::vector<std::vector<int> > > allHidden(R);

    int totNumIt = 0;
    while (true) {
        //if ((totNumIt & (totNumIt - 1)) == 0) {
            // Who knows what is best here... this runs if totNumIt is a power of two so
            // that it becomes more rare as time goes on
            allHidden.clear();
            // std::cout << "Generating hidden feasible hidden states randomly.\n";
            int tempCounter = 0;
            for (size_t r = 0; r < R; ++r) {
                size_t numIt=0;
                while (numIt <= C * H * std::max(H, O) / (R * (totTime - 1))) {  // This is so that we have enough counts for A[h][h'] and E[h][o]
                    std::vector<int> hidden = generate_feasible_hidden(obs[r].size(), obs[r]);
                    allHidden[r].push_back(hidden);
                    ++tempCounter;
                    ++numIt;
                    if ((tempCounter % 100) == 0) {
                        std::cout << "Counter: " << totNumIt << " " << r << " " << R << " " << numIt << " " << C * H * std::max(H, O) / (R * (totTime - 1)) << "\n";
                    }
                }
            }
        //}

        ++totNumIt;

        for (size_t r = 0; r < R; ++r) {
            fill(SStar[r].begin(), SStar[r].end(), 0.);
            fill(SStarCounter[r].begin(), SStarCounter[r].end(), 0);
            for (size_t h = 0; h < H; ++h) {
                fill(AStar[r][h].begin(), AStar[r][h].end(), 0.);
                fill(AStarCounter[r][h].begin(), AStarCounter[r][h].end(), 0);
                fill(EStar[r][h].begin(), EStar[r][h].end(), 0.);
                fill(EStarCounter[r][h].begin(), EStarCounter[r][h].end(), 0);
            }
        }

        for (size_t r = 0; r < R; ++r) {
            int T = obs[r].size();
            for (size_t i = 0; i < allHidden[r].size(); ++i) {
                double p = std::exp(hmm.logProb(obs[r], allHidden[r][i]));
                // double p = 1.;
                SStar[r][allHidden[r][i][0]] += p;
                ++SStarCounter[r][allHidden[r][i][0]];
                for (int t = 0; t < T - 1; ++t) {
                    AStar[r][allHidden[r][i][t]][allHidden[r][i][t + 1]] += p;
                    ++AStarCounter[r][allHidden[r][i][t]][allHidden[r][i][t + 1]];
                    EStar[r][allHidden[r][i][t]][obs[r][t]] += p;
                    ++EStarCounter[r][allHidden[r][i][t]][obs[r][t]];
                }
                EStar[r][allHidden[r][i][T - 1]][obs[r][T - 1]] += p;
                ++EStarCounter[r][allHidden[r][i][T - 1]][obs[r][T - 1]];
            }
        }

        // Normalize
        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                if (SStarCounter[r][h] == 0) {  // WEH - Isn't this always true?
                    SStar[r][h] = 0;
                }
                else {
                    SStar[r][h] /= SStarCounter[r][h];
                }
            }
            double SSum = std::accumulate(SStar[r].begin(), SStar[r].end(), 0.);
            for (size_t h = 0; h < H; ++h) {
                SStar[r][h] = SStar[r][h] / SSum;
            }
        }
        for (size_t h = 0; h < H; ++h) {
            double tempSum = 0.;
            for (size_t r = 0; r < R; ++r) {
                tempSum += SStar[r][h];
            }
            S[h] = tempSum / R;
        }
        hmm.setS(S);

        for (size_t r = 0; r < R; ++r) {
            for (size_t h1 = 0; h1 < H; ++h1) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    if (AStarCounter[r][h1][h2] == 0) {
                        AStar[r][h1][h2] = 0;
                    }
                    else {
                        AStar[r][h1][h2] = AStar[r][h1][h2] / AStarCounter[r][h1][h2];
                    }
                }
            }

            double ASum = 0;
            for (size_t h1 = 0; h1 < H; ++h1) {
                ASum += std::accumulate(AStar[r][h1].begin(), AStar[r][h1].end(), 0.);
            }
            for (size_t h1 = 0; h1 < H; ++h1) {
                for (size_t h2 = 0; h2 < H; ++h2) {
                    AStar[r][h1][h2] /= ASum;
                }
            }
        }
        double tol = 0.;
        std::vector<std::vector<double> > newA;
        newA.resize(H);
        for (size_t h1 = 0; h1 < H; ++h1) {
            newA[h1].resize(H);
            for (size_t h2 = 0; h2 < H; ++h2) {
                double tempSum = 0.;
                for (size_t r = 0; r < R; ++r) {
                    tempSum += AStar[r][h1][h2];
                }
                newA[h1][h2] = tempSum;
            }
            double ASum = std::accumulate(newA[h1].begin(), newA[h1].end(), 0.);
            for (size_t h2 = 0; h2 < H; ++h2) {
                if (ASum == 0) {
                    newA[h1][h2] = 1. / H;
                }
                else {
                    newA[h1][h2] = newA[h1][h2] / ASum;
                }
                tol = std::max(std::abs(A[h1][h2] - newA[h1][h2]), tol);
                A[h1][h2] = newA[h1][h2];
            }
        }
        hmm.setA(A);

        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t o = 0; o < O; ++o) {
                    if (EStarCounter[r][h][o] == 0) {
                        EStar[r][h][o] = 0;
                    }
                    else {
                        EStar[r][h][o] /= EStarCounter[r][h][o];
                    }
                }
            }

            double ESum = 0.;
            for (size_t h = 0; h < H; ++h) {
                ESum += std::accumulate(EStar[r][h].begin(), EStar[r][h].end(), 0.);
            }
            for (size_t h = 0; h < H; ++h) {
                for (size_t o = 0; o < O; ++o) {
                    EStar[r][h][o] /= ESum;
                }
            }
        }

        for (size_t h = 0; h < H; ++h) {
            for (size_t o = 0; o < O; ++o) {
                double tempSum = 0.;
                for (size_t r = 0; r < R; ++r) {
                    tempSum += EStar[r][h][o];
                }
                E[h][o] = tempSum;
            }
            double ESum = std::accumulate(E[h].begin(), E[h].end(), 0.);
            for (size_t o = 0; o < O; ++o) {
                if (ESum == 0) {
                    E[h][o] = 1. / O;
                }
                else {
                    E[h][o] = E[h][o] / ESum;
                }
            }
        }
        hmm.setE(E);

        std::cout << "Tolerance: " << totNumIt << " " << tol << " " << convergence_tolerance << std::endl;
        if (tol < convergence_tolerance) {
            break;
        std::cout << "Tolerance: " << totNumIt << " " << tol << std::endl;
        }
    }
}

void LearnStochastic::learn(const std::vector<std::vector<int> > &obs, const Options &options)
{
    double convergence_tolerance = 10E-6;
    unsigned int C = 10E4;
    process_options(options, convergence_tolerance, C);

    learn(obs, convergence_tolerance, C);
}

void LearnStochastic::learn(const std::vector<int> &obs, const Options &options)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    learn(newObs, options);
}

}  // namespace chmmpp

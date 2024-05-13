#include <cassert>
#include <iostream>
#include <set>
#include "chmmpp/learn/LearnStochastic.hpp"

namespace chmmpp {

namespace {

void process_options(const Options &options, double &convergence_tolerance, unsigned int &C, unsigned int& select)
{
    for (const auto &it : options.options) {
        if (it.first == "C") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp >= 0)
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
        else if (it.first == "select") {
            if (std::holds_alternative<int>(it.second)) {
                int tmp = std::get<int>(it.second);
                if (tmp >= 0)
                    select = static_cast<unsigned int>(tmp);
                else
                    std::cerr << "WARNING: 'select' option must be a non-negative integer" << std::endl;
            }
            else if (std::holds_alternative<unsigned int>(it.second)) {
                select = std::get<unsigned int>(it.second);
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
                    std::vector<int> hidden = generate_random_feasible_hidden(obs[r].size(), obs[r]);
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


void LearnStochastic::learn1(const std::vector<std::vector<int>>& observations,
                            const double convergence_tolerance, const int C)
{
    // For simplicity, this algorithm assumes that all observations have the same length
    size_t T = observations[0].size();
    for (auto& obs: observations)
        assert( T == obs.size() );
    size_t R = observations.size();

    auto A = hmm.getA();
    auto S = hmm.getS();
    auto E = hmm.getE();
    auto H = hmm.getH();    // # of hidden states
    auto O = hmm.getO();    // # of observed states

    std::vector<std::vector<double>> Sstar(R);
    std::vector<std::vector<std::vector<double>>> Astar(R);
    std::vector<std::vector<std::vector<double>>> Estar(R);
    for (size_t r = 0; r < R; ++r) {
        Sstar[r].resize(H);
        Astar[r].resize(H);
        Estar[r].resize(H);
        for (size_t h = 0; h < H; ++h) {
            Astar[r][h].resize(H);
            Estar[r][h].resize(O);
        }
    }

    //std::vector<size_t> Scount;
    //std::vector<size_t> Acount;
    //std::vector<std::vector<size_t>> Ecount;

#if 0
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
#endif

    std::set<std::vector<int>> hidden;

    // Maximum number of hidden sequences that we try to generate per major iteration
    size_t max_iterations = 100;
    size_t max_hidden_per_iteration = R*4;
    size_t max_trials = 10;

    int totNumIt = 0;
    while (totNumIt < max_iterations) {
        size_t num=0;
        // The sum of log_liklihoods for hidden states generated for all observations
        // WEH - Should this be monotonically increasing?
        double total_ll=0.0;

        // Find feasible hidden states with maximal likelihood
        for (auto& obs: observations) {
            auto [hid,log_likelihood] = generate_feasible_hidden(T, obs);
            total_ll += log_likelihood;
            auto results = hidden.insert(hid);
            if (results.second)
                num++;
        }
        std::cout << "Total Log-Liklihood: " << total_ll << std::endl;
        // Randomized trials
        for (size_t i=0; i<max_trials; ++i) {
            for (auto& obs: observations) {
                auto results = hidden.insert(generate_random_feasible_hidden(T, obs));
                if (results.second)
                    num++;
                if (num >= max_hidden_per_iteration) break;
                }
            if (num >= max_hidden_per_iteration) break;
            }    

        // Clear weighted parameters data
        for (size_t r = 0; r < R; ++r) {
            fill(Sstar[r].begin(), Sstar[r].end(), 0.);
            //fill(SStarCounter[r].begin(), SStarCounter[r].end(), 0);
            for (size_t h = 0; h < H; ++h) {
                fill(Astar[r][h].begin(), Astar[r][h].end(), 0.);
                //fill(AStarCounter[r][h].begin(), AStarCounter[r][h].end(), 0);
                fill(Estar[r][h].begin(), Estar[r][h].end(), 0.);
                //fill(EStarCounter[r][h].begin(), EStarCounter[r][h].end(), 0);
            }
        }

        std::cout << "Wsize: " << hidden.size() << std::endl;
        std::vector<double> w(hidden.size());       // Weights of each hidden state sequence

        for (size_t r = 0; r < R; ++r) {
            auto& obs = observations[r];

            // Reweight hidden states for these observations
            {
            double wsum=0.0;
            for (size_t i=0; auto& hid: hidden)
                wsum += w[i++] = std::exp(hmm.logProb(obs, hid));
            for (auto& val: w)
                val /= wsum;
            }

            // Collect weighted parameters
            for (size_t i=0; auto& hid: hidden) {
                auto p = w[i++];

                Sstar[r][hid[0]] += p;
                //++Scount[r][hid[0]];

                for (int t = 0; t < T - 1; ++t) {
                    Astar[r][hid[t]][hid[t + 1]] += p;
                    //++Acount[r][hid[i][t]][hid[t + 1]];
                }

                for (int t = 0; t < T; ++t) {
                    Estar[r][hid[t]][obs[t]] += p;
                    //++Ecount[r][hid[t]][obs[t]];
                }
            }
        }

#if 0
        // Normalize Sstar over possible hidden state values
        for (size_t r = 0; r < R; ++r) {
            for (size_t h = 0; h < H; ++h) {
                if (Scount[r][h] == 0)
                    Sstar[r][h] = 0;
                else
                    Sstar[r][h] /= Scount[r][h];
            }
            double SSum = std::accumulate(Sstar[r].begin(), Sstar[r].end(), 0.);
            for (size_t h = 0; h < H; ++h) {
                Sstar[r][h] = Star[r][h] / SSum;
            }
        }
#endif

        // Compute S
        for (size_t h = 0; h < H; ++h) {
            double tempSum = 0.;
            for (size_t r = 0; r < R; ++r) {
                tempSum += Sstar[r][h];
            }
            S[h] = tempSum / R;
        }
        hmm.setS(S);

#if 0
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
#endif

        double tol = 0.;
        std::vector<std::vector<double> > newA(H);
        for (size_t h1 = 0; h1 < H; ++h1)
            newA[h1].resize(H);

        for (size_t h1 = 0; h1 < H; ++h1) {
            for (size_t h2 = 0; h2 < H; ++h2) {
                double tempSum = 0.;
                for (size_t r = 0; r < R; ++r) {
                    tempSum += Astar[r][h1][h2];
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

#if 0
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
#endif

        for (size_t h = 0; h < H; ++h) {
            for (size_t o = 0; o < O; ++o) {
                double tempSum = 0.;
                for (size_t r = 0; r < R; ++r) {
                    tempSum += Estar[r][h][o];
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

        ++totNumIt;

        std::cout << "Tolerance: " << totNumIt << " " << tol << " " << convergence_tolerance << std::endl;
        if (tol < convergence_tolerance) {
            break;
        std::cout << "Tolerance: " << totNumIt << " " << tol << std::endl;
        hmm.print();
        }
    }
}

void LearnStochastic::learn(const std::vector<std::vector<int> > &obs, const Options &options)
{
    double convergence_tolerance = 10E-6;
    unsigned int C = 10E4;
    unsigned int select = 0;
    process_options(options, convergence_tolerance, C, select);

    if (select == 0)
        learn(obs, convergence_tolerance, C);
    else if (select == 1)
        learn1(obs, convergence_tolerance, C);
}

void LearnStochastic::learn(const std::vector<int> &obs, const Options &options)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    learn(newObs, options);
}

}  // namespace chmmpp

#include <cassert>
#include <iostream>
#include <set>
#ifdef WITH_COEK
#include <coek/util/io_utils.hpp>
#endif
#include "chmmpp/learn/LearnStochastic.hpp"
#include "../inference/inference.hpp" //For learn_hardEM

namespace chmmpp {

namespace {

void process_options(Options &options, double &convergence_tolerance, unsigned int &C, unsigned int& select, unsigned int& max_iterations, unsigned int& quiet)
{
    options.get_option("C", C);
    options.get_option("select", select);
    options.get_option("max_iterations", max_iterations);
    options.get_option("quiet", quiet);

    options.clear_option("C");
    options.clear_option("select");
    options.clear_option("max_iterations");
    options.clear_option("quiet");
}


}  // namespace

void LearnStochastic::set_seed(long int seed)
{
    std::random_device rand_dev;
    std::mt19937 myGenerator(rand_dev());
    generator = myGenerator;
    generator.seed(seed);
}

long int LearnStochastic::generate_seed()
{ return generator(); }

// Will work best/fastest if the sets of hidden states which satisfy the constraints
// This algorithm is TERRIBLE, I can't even get it to converge in a simple case with T = 10.
// This is currently the only learning algorithm we have for having a constraint oracle rather than
// ``simple'' constraints This also fails to work if we are converging towards values in the
// transition matrix with 0's (which is NOT uncommon)
void LearnStochastic::learn(const std::vector<std::vector<int> > &obs,
                            double convergence_tolerance, unsigned int C, unsigned int max_iterations)
{
    set_seed(hmm.get_seed());

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
                    std::vector<int> hidden = generate_random_feasible_hidden(obs[r].size(), obs[r], generate_seed());
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
                tol = std::max(std::fabs(A[h1][h2] - newA[h1][h2]), tol);
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
        if (totNumIt >= max_iterations)
            break;
        if (tol < convergence_tolerance) {
            break;
        }
    }
}


void LearnStochastic::learn1(const std::vector<std::vector<int>>& observations,
                            double convergence_tolerance, unsigned int C, unsigned int max_iterations, unsigned int quiet)
{
    //
    // Control parameters
    //

    size_t max_hidden_per_iteration = 2*observations.size();
    size_t max_trials = 2;
    size_t freq_revisit = 10;

    // For simplicity, this algorithm assumes that all observations have the same length
    size_t T = observations[0].size();
    for (auto& obs: observations)
        assert( T == obs.size() );
    size_t R = observations.size();

    set_seed(hmm.get_seed());
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

    std::vector<std::vector<int>> hcache(R);
    std::set<std::vector<int>> hidden;


    size_t wprev=-1;
    size_t ngen=0;
    int totNumIt = 0;
    while (true) {
        size_t num=0;
        // The sum of log_likelihoods for hidden states generated for all observations
        // WEH - Should this be monotonically increasing?
        double total_ll=0.0;

        if ((wprev == hidden.size()) and (totNumIt % freq_revisit != 0)) {
            // If the number of feasible hidden states hasn't changed, then our updated HMM parameters
            //      aren't likely to impact the prediction of the most likely hidden states.
            //      Hence, we just update the log-likelihood estimate except that we occassionally
            //      double-check that we haven't predicted different most likely hidden states.
            for (size_t r=0; r<R; ++r)
                total_ll += hmm.logProb(observations[r], hcache[r]);
        }
        else {
            wprev = hidden.size();

            // Find feasible hidden states with maximal likelihood
            size_t nobs=0;
            for (auto& obs: observations) {
                auto [hid,tmp] = generate_feasible_hidden(T, obs);
#if 0
                auto nz_ = count(hid.begin(), hid.end(), 0);
                if (nz_ != 10) {
                    std::cout
                    std::cout << "Hidden:       " << hid << std::endl;
                    std::cout << "Num Zeros:    " << nz_ << std::endl;
                }
#endif
                ngen++;
                double log_likelihood = hmm.logProb(obs,hid);
                if (std::fabs(tmp-log_likelihood) > 1e-3) {
                    std::cout << "WARNING: Differing estimates of log-likelihood Iteration=" << totNumIt << " Observation=" << nobs << " mip=" << tmp << " HMM=" << log_likelihood << std::endl;
#ifdef WITH_COEK
                    std::cout << "Observations: " << obs << std::endl;
                    std::cout << "Hidden:       " << hid << std::endl;
#endif
                    hmm.print();
                    }
                total_ll += log_likelihood;
                auto results = hidden.insert(hid);
                if (results.second)
                    num++;
                hcache[nobs] = hid;
                nobs++;
            }
            // Randomized trials
            for (size_t i=0; i<max_trials; ++i) {
                for (auto& obs: observations) {
                    auto results = hidden.insert(generate_random_feasible_hidden(T, obs, generate_seed()));
                    ngen++;
                    if (results.second)
                        num++;
                    if (num >= max_hidden_per_iteration) break;
                    }
                if (num >= max_hidden_per_iteration) break;
                }    
        }
        if (not quiet)
            std::cout << "Total Log-Liklihood: " << total_ll << std::endl;

        // Clear weighted parameters data
        for (size_t r = 0; r < R; ++r) {
            fill(Sstar[r].begin(), Sstar[r].end(), 0.);
            for (size_t h = 0; h < H; ++h) {
                fill(Astar[r][h].begin(), Astar[r][h].end(), 0.);
                fill(Estar[r][h].begin(), Estar[r][h].end(), 0.);
            }
        }

        if (not quiet) {
            std::cout << "Wsize: " << hidden.size() << std::endl;
            std::cout << "Num solutions generated: " << ngen << std::endl;
        }

        std::map<std::vector<int>,double> w;
        for (size_t r = 0; r < R; ++r) {
            auto& obs = observations[r];

            // Reweight hidden states for these observations
            {
#if 1
            // WEH - What justifies this bias?
            double wsum=0.0;
            for (auto& hid: hidden)
                wsum += w[hid] = std::exp(hmm.logProb(obs, hid));
            for (auto& val: w)
                val.second /= wsum;
#else
            for (auto& hid: hidden)
                w[hid] = 1.0/hidden.size();
#endif
            }

#if 0
            if ((r == 0) and (debug)) {
                std::cout << "Obs: " << obs << std::endl;
                for (auto& hid: hidden)
                    std::cout << w[hid] << " : " << hid << std::endl;
                }
#endif

            // Collect weighted parameters
            for (auto& hid: hidden) {
                auto p = w[hid];

                Sstar[r][hid[0]] += p;

                for (int t = 0; t < T - 1; ++t) {
                    Astar[r][hid[t]][hid[t + 1]] += p;
                }

                for (int t = 0; t < T; ++t) {
                    Estar[r][hid[t]][obs[t]] += p;
                }
            }
        }

        // Compute S
        for (size_t h = 0; h < H; ++h) {
            double tempSum = 0.;
            for (size_t r = 0; r < R; ++r) {
                tempSum += Sstar[r][h];
            }
            S[h] = tempSum / R;
        }
        hmm.setS(S);

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
                tol = std::max(std::fabs(A[h1][h2] - newA[h1][h2]), tol);
                A[h1][h2] = newA[h1][h2];
            }
        }
        hmm.setA(A);

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

        if (not quiet) {
            std::cout << "Iteration= " << totNumIt << " Tolerance= " << tol << std::endl;
            std::cout << std::endl;
            hmm.print();
        }

        if (totNumIt >= max_iterations)
            break;
        if (tol < convergence_tolerance)
            break;
        ++totNumIt;
    }
}

void LearnStochastic::learn(const std::vector<std::vector<int> > &obs, Options &options)
{
    double convergence_tolerance = 10E-6;
    unsigned int C = 10E4;
    unsigned int max_iterations = 1000;
    unsigned int select = 0;
    unsigned int quiet = 1;
    process_options(options, convergence_tolerance, C, select, max_iterations, quiet);

    if (select == 0)
        learn(obs, convergence_tolerance, C, max_iterations);
    else if (select == 1)
        learn1(obs, convergence_tolerance, C, max_iterations, quiet);
}

void LearnStochastic::learn(const std::vector<int> &obs, Options &options)
{
    std::vector<std::vector<int> > newObs;
    newObs.push_back(obs);
    learn(newObs, options);
}

}  // namespace chmmpp

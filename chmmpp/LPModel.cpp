#include <cmath>
#include "LPModel.hpp"

namespace chmmpp {

void LPModel::initialize(const HMM& hmm, const std::vector<int>& observations)
{
    clear();

    Tmax = observations.size();  // # of time steps in the observations
    N = hmm.getH();              // # of hidden states

    const auto& start_probs = hmm.getS();     // map from 0..N-1 to a probability value in 0..1
    const auto& emission_probs = hmm.getE();  // emission_probs[i][k] - probability that output k is
                                              // generated when in hidden state i
    const auto& trans_mat = hmm.getA();       // trans_mat[i][j] - probability of transitioning from
                                              // hidden state i to hidden state j

    for (auto t : coek::range(Tmax)) {
        if (t == 0) {
            for (auto i : coek::range(N)) {
                auto tmp = start_probs[i] * emission_probs[i][observations[t]];
                if (tmp > 0)
                    G[{-1, -1, i}] = log(tmp);
                else
                    FF.insert({-1, -1, i});
            }
        }
        else {
            for (auto a : coek::range(N))
                for (auto b : coek::range(N)) {
                    auto val = trans_mat[a][b];
                    if (val > 0) F.insert({a, b});
                    auto tmp = val * emission_probs[b][observations[t]];
                    if (tmp > 0)
                        G[{t - 1, a, b}] = log(tmp);
                    else
                        FF.insert({t - 1, a, b});
                }
        }
    }

    // E = []  # (t,a,b) where
    //           (t, -1,  i) when t==-1
    //           (t,  a,  b) when t>=0 and t<Tmax
    //           (t,  i, -2) when t==Tmax

    for (auto i : coek::range(N)) E.push_back({-1, -1, i});
    for (auto t : coek::range(Tmax))
        for (auto& g : F) E.push_back({t, std::get<0>(g), std::get<1>(g)});
    for (auto i : coek::range(N)) E.push_back({Tmax, i, -2});

    if (y_binary) {
        for (auto& e : E) y[e] = coek::variable().bounds(0, 1).within(coek::Boolean);
    }
    else {
        for (auto& e : E) y[e] = coek::variable().bounds(0, 1);
    }

    // flow constraints
    for (auto t : coek::range(Tmax + 1)) {
        for (auto b : coek::range(N)) {
            auto lhs = coek::expression();
            auto rhs = coek::expression();
            if (t == 0)
                lhs = y[{t - 1, -1, b}];
            else {
                for (auto a : coek::range(N))
                    if (not(F.find({a, b}) == F.end())) lhs += y[{t - 1, a, b}];
            }
            if (t == Tmax)
                rhs = y[{t, b, -2}];
            else {
                for (auto aa : coek::range(N))
                    if (not(F.find({b, aa}) == F.end())) rhs += y[{t, b, aa}];
            }
            model.add(lhs == rhs);
        }
    }

    // flow_start
    {
        auto lhs = coek::expression();
        for (auto b : coek::range(N)) lhs += y[{-1, -1, b}];
        model.add(lhs == 1);
    }

    // flow end
    {
        auto lhs = coek::expression();
        for (auto a : coek::range(N)) lhs += y[{Tmax, a, -2}];
        model.add(lhs == 1);
    }

    // objective
    {
        auto O = coek::expression();
        for (auto& ff : FF) O += y[ff];
        for (auto& g : G) log_likelihood_expr += g.second * y[g.first];
        model.add_objective(log_likelihood_expr - pow(10.0, 6) * O).sense(model.maximize);
    }

    if (not keep_data) {
        E.clear();
        F.clear();
        FF.clear();
        G.clear();
    }
}

void LPModel::optimize(double& log_likelihood, std::vector<int>& hidden_states)
{
    coek::Solver solver(solver_name);
    if (not solver.available()) std::cout << "Error setting up solver " + solver_name << std::endl;

    auto status = solver.solve(model);
    if (status)
        std::cout << "Error executing linear programming solver: " + std::to_string(status)
                  << std::endl;

    collect_solution(hidden_states);
    log_likelihood = log_likelihood_expr.value();
}

void LPModel::collect_solution(std::vector<int>& hidden_states)
{
    for (auto& it : y) {
        if (it.second.value() > 0) {
            auto& [t, a, b] = it.first;
            size_t i = static_cast<size_t>(t + 1);
            if (i < Tmax) hidden_states[i] = b;
        }
    }
}

}  // namespace chmmpp
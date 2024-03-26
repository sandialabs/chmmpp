#pragma once

#include <string>
#include <map>
#include <tuple>
#include <coek/coek.hpp>
#include "chmmpp/HMM.hpp"

namespace chmmpp {

class LPModel {
   protected:
    coek::Model model;

    size_t Tmax;
    size_t N;

    std::map<std::tuple<int, int, int>, coek::Variable> y;
    coek::Expression log_likelihood_expr;

    std::vector<std::tuple<int, int, int>> E;
    std::map<std::tuple<int, int, int>, double> G;
    std::set<std::tuple<int, int>> F;
    std::set<std::tuple<int, int, int>> FF;

   public:
    // Configuration options
    std::string solver_name = "gurobi";
    bool y_binary = false;
    bool keep_data = true;

   public:
    virtual void initialize(const HMM& hmm, const std::vector<int>& observations);

    virtual void optimize(double& log_likelihood, std::vector<int>& hidden_states);

    virtual void collect_solution(std::vector<int>& hidden_states);

    virtual void clear();
};

}  // namespace chmmpp

#include "syntheticCitationHMM.hpp"
#ifdef WITH_COEK
#    include <coek/util/io_utils.hpp>
#    include <chmmpp/inference/LPModel.hpp>
#endif

namespace chmmpp {

// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------

class InferenceModel : public LPModel {
   public:
    // virtual void set_options(const Options& options);

    void initialize(const syntheticCitationHMM& nzhmm, const std::vector<int>& observations);

    // void optimize(double& log_likelihood, std::vector<int>& hidden_states);

    void collect_solution(std::vector<int>& hidden_states);

    virtual void print_solution();
};


class LearningModel : public InferenceModel {
   public:
    std::vector<double> unconstrained_hidden;
    std::map<std::pair<size_t,size_t>,coek::Variable> z;

    void initialize(syntheticCitationHMM& nzhmm, const std::vector<int>& observations, const std::vector<int>& hidden);
    void print_solution();
};

class Generator_MIP_SyntheticCitation : public Generator_Base {
   public:
    virtual std::vector<std::vector<std::vector<int>>> operator()(
        HMM &hmm, const std::vector<std::vector<int>>& obs
    ) const;
};

} // namespace chmmpp

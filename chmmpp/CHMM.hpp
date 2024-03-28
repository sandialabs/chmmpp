#pragma once

#include <functional>
#include <chmmpp/HMM.hpp>

namespace chmmpp {

//
// A base class that supports various methods for constrained inference.
//
class CHMM : public Options
{
public:

    HMM hmm;
    std::function<bool(std::vector<int>&)> constraintOracle;

public:

    void initialize(const HMM& _hmm)
        {hmm = _hmm;}

    void initialize(const std::vector<std::vector<double> > &inputA,
                    const std::vector<double> &inputS,
                    const std::vector<std::vector<double> > &inputE, long int seed)
        {hmm.initialize(inputA, inputS, inputE, seed);}

    virtual void initialize_from_file(const std::string &json_filename)
        {hmm.initialize_from_file(json_filename);}

    virtual void initialize_from_string(const std::string &json_string)
        {hmm.initialize_from_string(json_string);}

    // aStar using the constraintOracle object
    void aStar(const std::vector<int> &observations, std::vector<int> &hidden_states, double &logProb);

    // aStar generating multiple solutions
    void aStarMult(const std::vector<int> &observations, std::vector<std::vector<int>> &hidden_states, std::vector<double> &logProb, const int numSolns);

    // Optimize using an mixed-integer programming formulation that expresses application constraints
    virtual void mip_map_inference(const std::vector<int> &observations, std::vector<int> &hidden_states, double &logProb);

    virtual double logProb(const std::vector<int> obs, const std::vector<int> guess) const
        {return hmm.logProb(obs, guess);}

    Options& get_options()
        {return hmm.get_options();}

    const Options& get_options() const
        {return hmm.get_options();}


};

} // namespace chmmpp

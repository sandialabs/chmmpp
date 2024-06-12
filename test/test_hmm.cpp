#include "catch2/catch_test_macros.hpp"

#include <chmmpp/HMM.hpp>

TEST_CASE("hmm1", "[hmm]")
{
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1},{1,0}};

    SECTION("seed")
    {
        chmmpp::HMM hmm;
        hmm.set_seed(123456789);
        REQUIRE(hmm.get_seed() == 123456789);
    }

    SECTION("constructor")
    {
        chmmpp::HMM hmm(A,S,E);
        REQUIRE(hmm.getH() == 2);   // # hidden states
        REQUIRE(hmm.getO() == 2);   // # observed states
    }

    SECTION("initialize")
    {
        chmmpp::HMM hmm;
        REQUIRE(hmm.getH() == 0);   // # hidden states
        REQUIRE(hmm.getO() == 0);   // # observed states

        hmm.initialize(A,S,E);
        REQUIRE(hmm.getH() == 2);   // # hidden states
        REQUIRE(hmm.getO() == 2);   // # observed states
    }

    SECTION("getAEntry")
    {
        chmmpp::HMM hmm(A,S,E);
        REQUIRE(hmm.getAEntry(0,0) == 0);
        REQUIRE(hmm.getAEntry(0,1) == 1);
        REQUIRE(hmm.getAEntry(1,0) == 1);
        REQUIRE(hmm.getAEntry(1,1) == 0);
    }

    SECTION("getSEntry")
    {
        chmmpp::HMM hmm(A,S,E);
        REQUIRE(hmm.getSEntry(0) == 1);
        REQUIRE(hmm.getSEntry(1) == 0);
    }

    SECTION("getEEntry")
    {
        chmmpp::HMM hmm(A,S,E);
        REQUIRE(hmm.getEEntry(0,0) == 0);
        REQUIRE(hmm.getEEntry(0,1) == 1);
        REQUIRE(hmm.getEEntry(1,0) == 1);
        REQUIRE(hmm.getEEntry(1,1) == 0);
    }

    SECTION("run")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs(5);
        std::vector<int> hidden(5);
        hmm.run(5, obs, hidden);
        std::vector<int> obs_ = {1,0,1,0,1};
        std::vector<int> hidden_ = {0,1,0,1,0};
        REQUIRE(obs == obs_);
        REQUIRE(hidden == hidden_);
    }

    SECTION("log-likelihood")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        std::vector<int> hidden = {0,1,0,1,0};
        double log_likelihood = hmm.logProb(obs, hidden);
        REQUIRE(log_likelihood == 0.0);
    }

    SECTION("viterbi")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        std::vector<int> hidden;
        double log_likelihood=0.0;
        hmm.viterbi(obs, hidden, log_likelihood);
        REQUIRE(log_likelihood == 0.0);
    }

    SECTION("aStar")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        std::vector<int> hidden;
        double log_likelihood=0.0;
        hmm.aStar(obs, hidden, log_likelihood);
        REQUIRE(log_likelihood == 0.0);
    }

    SECTION("lp")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        std::vector<int> hidden;
        double log_likelihood=0.0;
        hmm.set_option("debug",false);
        hmm.lp_map_inference(obs, hidden, log_likelihood);
        REQUIRE(log_likelihood == 0.);
    }

    SECTION("estimate_hmm") {
        chmmpp::HMM hmm(A,S,E); //A,S,E don't matter here just their sizes.
        std::vector<int> hid{0,0,0,1,1};
        std::vector<int> obs{0,1,0,1,0};
        hmm.estimate_hmm(obs,hid);
        std::vector<std::vector<double>> _A = {{2./3.,1./3.},{0.,1.}};
        std::vector<double> _S = {1.,0.};
        std::vector<std::vector<double>> _E = {{2./3.,1./3.},{1./2.,1./2.}};
        REQUIRE(hmm.getA() == _A);
        REQUIRE(hmm.getS() == _S);
        REQUIRE(hmm.getE() == _E);
    }

    //TODO make more than one test for this
    SECTION("baum_welch single observation") {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        hmm.baum_welch(obs);
        REQUIRE(hmm.getA() == A);
        REQUIRE(hmm.getS() == S);
        REQUIRE(hmm.getE() == E);
    }

    SECTION("baum_welch multiple observation") {
        chmmpp::HMM hmm(A,S,E);
        std::vector<std::vector<int>> obs = {{1},{1,0,1,0,1},{1,0}};
        hmm.baum_welch(obs);
        REQUIRE(hmm.getA() == A);
        REQUIRE(hmm.getS() == S);
        REQUIRE(hmm.getE() == E);
    }

}

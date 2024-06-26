#include <cmath>
#include "catch2/catch_test_macros.hpp"
#include <chmmpp/HMM.hpp>
#include <chmmpp/util/math.hpp>

#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)

TEST_CASE("hmm_errors", "[hmm]")
{
    SECTION("empty_E")
    {
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("inconsistent_A_size")
    {
    std::vector<std::vector<double>> A = {{0,1,0},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("inconsistent_E_size")
    {
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1,0},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("negative_A_value")
    {
    std::vector<std::vector<double>> A = {{0,-1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("rows_of_A_sum_to_one")
    {
    std::vector<std::vector<double>> A = {{0,0.5},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("negative_S_value")
    {
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {-1,0};
    std::vector<std::vector<double>> E = {{0,1},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("negative_E_value")
    {
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,-1},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("rows_of_E_sum_to_one")
    {
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,0.5},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

    SECTION("S_sum_to_one")
    {
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {0.5,0};
    std::vector<std::vector<double>> E = {{0,1.0},{1,0}};

    chmmpp::HMM hmm;
    REQUIRE_THROWS(hmm.initialize(A,S,E));
    }

}

TEST_CASE("hmm1", "[hmm]")
{
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1},{1,0}};

    // API tests

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

    SECTION("randomize")
    {
        chmmpp::HMM hmm(2,2);
        hmm.set_seed(123456789);
        hmm.randomize();
        auto A = hmm.getA();
        auto E = hmm.getE();
        auto S = hmm.getS();
        hmm.initialize(A,S,E);
    }

    // Model operation tests

    SECTION("run")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs(5);
        std::vector<int> hidden(5);
        hmm.run(obs, hidden);
        std::vector<int> obs_ = {1,0,1,0,1};
        std::vector<int> hidden_ = {0,1,0,1,0};
        REQUIRE(obs == obs_);
        REQUIRE(hidden == hidden_);
    }

    SECTION("generate_hidden")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        auto hidden = hmm.generateHidden(obs);
        std::vector<int> hidden_ = {0,1,0,1,0};
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

    // Inference tests

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
        hmm.set_option("debug",true);
        double log_likelihood=0.0;
#if 0
        hmm.lp_map_inference(obs, hidden, log_likelihood);
#endif
        REQUIRE(log_likelihood == 0.);
    }

    SECTION("estimate_hmm") {
        chmmpp::HMM hmm(2,2);
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

    SECTION("baum_welch single observation") {
        chmmpp::HMM test(A,S,E);
        test.print();
        chmmpp::HMM hmm(2,2);
        std::vector<int> obs = {1,0,1,0,1};
        hmm.set_seed(123456789);
        hmm.randomize();
        hmm.baum_welch(obs);
        hmm.print();
        auto [flagA,strA] = chmmpp::compare(hmm.getA(), A, 1e-7);
        REQUIRE_MESSAGE(flagA,strA);
        auto [flagS,strS] = chmmpp::compare(hmm.getS(), S, 1e-7);
        REQUIRE_MESSAGE(flagS,strS);
        auto [flagE,strE] = chmmpp::compare(hmm.getE(), E, 1e-7);
        REQUIRE_MESSAGE(flagE,strE);
    }

    SECTION("baum_welch multiple observation") {
        chmmpp::HMM test(A,S,E);
        test.print();
        chmmpp::HMM hmm(2,2);
        std::vector<std::vector<int>> obs = {{1},{1,0,1,0,1},{1,0}};
        hmm.set_seed(123456789);
        hmm.randomize();
        hmm.baum_welch(obs);
        hmm.print();
        auto [flagA,strA] = chmmpp::compare(hmm.getA(), A, 1e-7);
        REQUIRE_MESSAGE(flagA,strA);
        auto [flagS,strS] = chmmpp::compare(hmm.getS(), S, 1e-7);
        REQUIRE_MESSAGE(flagS,strS);
        auto [flagE,strE] = chmmpp::compare(hmm.getE(), E, 1e-7);
        REQUIRE_MESSAGE(flagE,strE);
    }

}


TEST_CASE("hmm2", "[hmm]")
{
    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1},{0.25,0.75}};

    // API tests

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
        REQUIRE(hmm.getEEntry(1,0) == 0.25);
        REQUIRE(hmm.getEEntry(1,1) == 0.75);
    }

    SECTION("randomize")
    {
        chmmpp::HMM hmm(2,2);
        hmm.set_seed(123456789);
        hmm.randomize();
        auto A = hmm.getA();
        auto E = hmm.getE();
        auto S = hmm.getS();
        hmm.initialize(A,S,E);
    }

    // Model operation tests

    SECTION("run")
    {
        chmmpp::HMM hmm(A,S,E);
        hmm.set_seed(123456789);
        std::vector<int> obs(5);
        std::vector<int> hidden(5);
        hmm.run(obs, hidden);
        std::vector<int> obs_ = {1,0,1,1,1};
        std::vector<int> hidden_ = {0,1,0,1,0};
        REQUIRE(obs == obs_);
        REQUIRE(hidden == hidden_);
    }

    SECTION("generate_hidden")
    {
        chmmpp::HMM hmm(A,S,E);
        hmm.set_seed(123456789);
        std::vector<int> obs = {1,0,1,1,1};
        auto hidden = hmm.generateHidden(obs);
        std::vector<int> hidden_ = {0,1,0,1,0};
        REQUIRE(hidden == hidden_);
    }

    SECTION("log-likelihood")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs =    {1,0,1,1,1};
        std::vector<int> hidden = {0,1,0,1,0};
        double log_likelihood = hmm.logProb(obs, hidden);
        REQUIRE(exp(log_likelihood) == 0.25*0.75);
    }

    SECTION("viterbi")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        std::vector<int> hidden;
        std::vector<int> hidden_ = {0,1,0,1,0};
        double log_likelihood=0.0;
        hmm.viterbi(obs, hidden, log_likelihood);
        REQUIRE(log_likelihood <= 0);
        REQUIRE(exp(log_likelihood) == 0.25*0.25);
        REQUIRE(hidden == hidden_);
    }

    SECTION("aStar")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        std::vector<int> hidden;
        std::vector<int> hidden_ = {0,1,0,1,0};
        double log_likelihood=0.0;
        hmm.aStar(obs, hidden, log_likelihood);
        REQUIRE(log_likelihood <= 0);
        REQUIRE(exp(log_likelihood) == 0.25*0.25);
        REQUIRE(hidden == hidden_);
    }

    SECTION("lp")
    {
        chmmpp::HMM hmm(A,S,E);
        std::vector<int> obs = {1,0,1,0,1};
        std::vector<int> hidden;
        std::vector<int> hidden_ = {0,1,0,1,0};
        double log_likelihood=0.0;
        hmm.set_option("debug",true);
#if 0
        hmm.lp_map_inference(obs, hidden, log_likelihood);
#endif
        REQUIRE(log_likelihood == 0.0);
    }

}

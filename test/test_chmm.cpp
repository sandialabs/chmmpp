#include "catch2/catch_test_macros.hpp"

#include <chmmpp/CHMM.hpp>

TEST_CASE("chmm1", "[chmm]")
{

    std::vector<std::vector<double>> A = {{0,1},{1,0}};
    std::vector<double> S = {1,0};
    std::vector<std::vector<double>> E = {{0,1},{1,0}};

    SECTION("initialize")
    {
        chmmpp::CHMM chmm1;
        chmmpp::CHMM chmm2;
        chmmpp::HMM hmm;

        hmm.initialize(A,S,E);
        chmm1.initialize(A,S,E);
        chmm2.initialize(hmm);
        REQUIRE(chmm1.hmm.getA() == chmm2.hmm.getA());
        REQUIRE(chmm1.hmm.getS() == chmm2.hmm.getS());
        REQUIRE(chmm1.hmm.getE() == chmm2.hmm.getE());
    } 

    SECTION("seed")
    {
        chmmpp::CHMM chmm;
        chmm.initialize(A,S,E);
        chmm.set_seed(123456789);
        REQUIRE(chmm.get_seed() == 123456789);
    }
}

#include "catch2/catch_test_macros.hpp"

unsigned int Factorial(unsigned int number)
{
    return number <= 1 ? number : Factorial(number - 1) * number;
}

TEST_CASE("Factorial Test", "[factorial]")
{
    SECTION("values")
    {
        REQUIRE(Factorial(2) == 2);
        REQUIRE(Factorial(3) == 6);
        REQUIRE(Factorial(4) == 24);
    }
}

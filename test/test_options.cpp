#include "catch2/catch_test_macros.hpp"
#include <chmmpp/util/Options.hpp>
#include <cmath>

TEST_CASE("options", "[util]")
{
    SECTION("set")
    {
    chmmpp::Options options;
    REQUIRE(options.num_options() == 0);
    options.set_option("foo", 1);
    REQUIRE(options.num_options() == 1);
    }

    SECTION("get")
    {
    chmmpp::Options options;
    options.set_option("foo", 1);
    int foo=0;
    options.get_option("foo", foo);
    REQUIRE(foo == 1);
    }

    SECTION("get_missing")
    {
    chmmpp::Options options;
    options.set_option("foo", 1);
    int bar=0;
    options.get_option("bar", bar);
    REQUIRE(bar == 0);
    }

    SECTION("get_wrong type")
    {
    chmmpp::Options options;
    options.set_option("foo", 1.5);
    int foo=0;
    options.get_option("foo", foo);
    REQUIRE(foo == 0);
    }

    SECTION("get unsigned 1")
    {
    chmmpp::Options options;
    unsigned int foo=1;
    options.set_option("foo", foo);
    unsigned int foo_=0;
    options.get_option("foo", foo_);
    REQUIRE(foo_== 1);
    }

    SECTION("get unsigned 2")
    {
    chmmpp::Options options;
    int foo=1;
    options.set_option("foo", foo);
    unsigned int foo_=0;
    options.get_option("foo", foo_);
    REQUIRE(foo_== 1);
    }

    SECTION("get unsigned 3")
    {
    chmmpp::Options options;
    int foo=-1;
    options.set_option("foo", foo);
    unsigned int foo_=0;
    options.get_option("foo", foo_);
    REQUIRE(foo_== 0);
    }

    SECTION("get_options")
    {
    chmmpp::Options options;
    options.set_option("foo", 1);
    auto& options_ = options.get_options();
    REQUIRE(options_.num_options() == 1);
    }

    SECTION("clear_options")
    {
    chmmpp::Options options;
    options.set_option("foo", 1);
    options.clear_options();
    REQUIRE(options.num_options() == 0);
    }

    SECTION("clear_option")
    {
    chmmpp::Options options;
    options.set_option("foo", 1);
    options.set_option("bar", 2);
    options.clear_option("foo");
    REQUIRE(options.num_options() == 1);
    }

}

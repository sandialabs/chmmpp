
find_package(Catch2 REQUIRED)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_INSTALL_PREFIX}/include)

if (with_boost)
    find_package(Boost REQUIRED)
endif()
if (with_fmtlib)
    find_package(fmt REQUIRED)
endif()

SET(sources
    runner.cpp
    test_foo.cpp
    )

set(chmmpp_test_libraries)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(BEFORE ${chmmpp_include_directories})
LINK_DIRECTORIES(${chmmpp_link_directories})

enable_testing()

# runner
add_executable(runner ${sources})
TARGET_LINK_LIBRARIES(runner PUBLIC chmmpp Catch2::Catch2 ${chmmpp_link_libraries} ${chmmpp_test_libraries})
TARGET_COMPILE_OPTIONS(runner PUBLIC ${chmmpp_compile_options})
target_compile_definitions(runner PUBLIC CHMMPP_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

include(CTest)
include(Catch)
catch_discover_tests(runner)

# testing
add_custom_target(test COMMAND ${CMAKE_CTEST_COMMAND})

if (gcovr_FOUND)
    add_custom_target(test_gcovr COMMAND ${GCOVR_EXECUTABLE} -d -r ${chmmpp_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/..)
endif()


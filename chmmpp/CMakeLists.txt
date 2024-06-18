##################### Build Shared Library  #####################

set(chmmpp_compile_options)
set(chmmpp_link_libraries)
set(chmmpp_include_directories)
set(chmmpp_link_directories)

SET(sources
    util/vectorhash.cpp
    util/Options.cpp
    HMM.cpp
    CHMM.cpp
    inference/lp_map.cpp
    learn/estimate_hmm.cpp
    learn/log_likelihood_estimate.cpp
    )

if (with_coek)
    find_package(coek REQUIRED)
    list(APPEND sources
            inference/LPModel.cpp
            )
    list(APPEND chmmpp_link_libraries "coek::coek")
    list(APPEND chmmpp_compile_options "-DWITH_COEK")
endif()

if (with_boost)
    find_package(Boost REQUIRED)
    list(APPEND sources 
            inference/HMM_inference.cpp
            inference/aStar.cpp
            inference/viterbi.cpp
            inference/aStar_oracle.cpp

	    learn/learn_unconstrained.cpp
	    learn/LearnStochastic.cpp
            learn/learn_semisupervised_hardEM.cpp
            learn/learn_batch.cpp
            )
    list(APPEND chmmpp_link_libraries "Boost::boost")
    list(APPEND chmmpp_compile_options "-DWITH_BOOST")
endif()

# gcov
if(with_gcov)
    list(APPEND chmmpp_compile_options -fprofile-arcs -ftest-coverage -g)
    list(APPEND chmmpp_link_libraries gcov)
endif()

# gprof
if (MSVC)
else()
    if(${with_gprof})
        list(APPEND chmmpp_compile_options "-pg")
    else()
        list(APPEND chmmpp_compile_options "-Wall;-O3")
    endif()
endif()

#
# Build object library
#
if (build_shared_libs)
    add_library(chmmpp SHARED ${sources})
    message("-- Building shared libchmmpp library")
    set_target_properties(chmmpp PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
else()
    add_library(chmmpp STATIC ${sources})
    message("-- Building static libchmmpp library")
endif()
add_library(chmmpp::chmmpp ALIAS chmmpp)

#message("HERE ${chmmpp_include_directories}")
#message("HERE ${chmmpp_compile_options}")
#message("HERE ${chmmpp_link_libraries}")
#message("HERE ${chmmpp_link_directories}")

target_compile_options(chmmpp PUBLIC ${chmmpp_compile_options})
target_include_directories(chmmpp
                           PUBLIC
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
                           )
target_include_directories(chmmpp
                           PUBLIC
                           ${chmmpp_include_directories})
target_link_libraries(chmmpp PUBLIC ${chmmpp_link_libraries})
target_link_directories(chmmpp PUBLIC ${chmmpp_link_directories})
set_property(TARGET chmmpp PROPERTY INTERFACE_LINK_LIBRARIES ${chmmpp_link_libraries})

#
# make install
#
install(TARGETS chmmpp
        EXPORT chmmppTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
       )
if (WIN32)
    set_target_properties(chmmpp PROPERTIES OUTPUT_NAME libchmmpp)
endif()

install(FILES
        chmmpp.hpp
        HMM.hpp
        CHMM.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chmmpp
        )
install(FILES
        Options.hpp
        vectorhash.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chmmpp/util
        )

#
# Push the compile options to the parent scope
#
set(chmmpp_compile_options ${chmmpp_compile_options} PARENT_SCOPE)
set(chmmpp_link_libraries ${chmmpp_link_libraries} PARENT_SCOPE)
set(chmmpp_include_directories ${chmmpp_include_directories} PARENT_SCOPE)
set(chmmpp_link_directories ${chmmpp_link_directories} PARENT_SCOPE)




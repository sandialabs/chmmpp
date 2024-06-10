project (CHMM_Examples)

set(EXAMPLES 
    simpleLearn.cpp
    simpleLearnMC.cpp
	tourInfer.cpp
    tourInferMultiple.cpp
)

set(numZeros
    numZeros/numZeros_aStar.cpp
    numZeros/numZeros_mip.cpp
    numZeros/numZerosHMM.cpp
    numZeros/numZeros_learn.cpp
)

set(syntheticCitation
    syntheticCitation/syntheticCitationHMM.cpp
)

set(citation
    citation/citation_learn.cpp
    citation/citation_mip.cpp
    citation/citationHMM.cpp
    citation/citation_aStar.cpp
    citation/readFile.cpp
)

#numZeros
add_executable(numzeros_infer numZeros/simpleInfer.cpp ${numZeros})
target_link_libraries(numzeros_infer PRIVATE chmmpp::chmmpp)

add_executable(numzeros_minfer numZeros/multInfer.cpp ${numZeros})
target_link_libraries(numzeros_minfer PRIVATE chmmpp::chmmpp)

add_executable(numzeros_learn numZeros/simpleLearn.cpp ${numZeros})
target_link_libraries(numzeros_learn PRIVATE chmmpp::chmmpp)

add_executable(numzeros_learn_test numZeros/learnTest.cpp ${numZeros})
target_link_libraries(numzeros_learn_test PRIVATE chmmpp::chmmpp)


#syntheticCitation
add_executable(syntheticCitation_learn syntheticCitation/simpleLearn.cpp ${syntheticCitation})
target_link_libraries(syntheticCitation_learn PRIVATE chmmpp::chmmpp)

add_executable(syntheticCitation_learn_test syntheticCitation/learnTest.cpp ${syntheticCitation})
target_link_libraries(syntheticCitation_learn_test PRIVATE chmmpp::chmmpp)


#citation
add_executable(citation_learn citation/simpleLearn.cpp ${citation})
target_link_libraries(citation_learn PRIVATE chmmpp::chmmpp)

#Go through all files in examples and add an executable for each
foreach( exampleFile ${EXAMPLES} )
    # Cut off the file extension and directory path
    get_filename_component( exampleName ${exampleFile} NAME_WE )
    
    add_executable( ${exampleName} ${exampleFile} )
    
    target_link_libraries(${exampleName}
        PRIVATE
        chmmpp::chmmpp
    )
endforeach( exampleFile ${EXAMPLES} )

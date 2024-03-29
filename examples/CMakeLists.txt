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

add_executable(simpleInfer numZeros/simpleInfer.cpp ${numZeros})
target_link_libraries(simpleInfer PRIVATE chmmpp::chmmpp)

add_executable(multInfer numZeros/multInfer.cpp ${numZeros})
target_link_libraries(multInfer PRIVATE chmmpp::chmmpp)

add_executable(numzeros_learn numZeros/simpleLearn.cpp ${numZeros})
target_link_libraries(numzeros_learn PRIVATE chmmpp::chmmpp)

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

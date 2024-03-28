project (CHMM_Examples)

#Go through all files in examples and add an executable for each
set(EXAMPLES 
    simpleInferMultiple.cpp
    simpleLearn.cpp
    simpleLearnMC.cpp
	tourInfer.cpp
    tourInferMultiple.cpp
)

add_executable(simpleInfer numZeros/simpleInfer.cpp numZeros/numZeros_aStar.cpp numZeros/numZeros_mip.cpp numZeros/numZerosHMM.cpp)
target_link_libraries(simpleInfer PRIVATE chmmpp::chmmpp)

foreach( exampleFile ${EXAMPLES} )
    # Cut off the file extension and directory path
    get_filename_component( exampleName ${exampleFile} NAME_WE )
    
    add_executable( ${exampleName} ${exampleFile} )
    
    target_link_libraries(${exampleName}
        PRIVATE
        chmmpp::chmmpp
    )
endforeach( exampleFile ${EXAMPLES} )

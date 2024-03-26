project (CHMM_Examples)

#Go through all files in examples and add an executable for each
set(EXAMPLES 
	simpleInfer.cpp simpleInferMultiple.cpp simpleLearn.cpp simpleLearnMC.cpp
	tourInfer.cpp tourInferMultiple.cpp
)


foreach( exampleFile ${EXAMPLES} )
    # Cut off the file extension and directory path
    get_filename_component( exampleName ${exampleFile} NAME_WE )
    
    add_executable( ${exampleName} ${exampleFile} )
    
    target_link_libraries(${exampleName} #hmm.h
        PRIVATE
        chmmpp::chmmpp
    )

endforeach( exampleFile ${EXAMPLES} )

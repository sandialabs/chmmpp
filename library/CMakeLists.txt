project(CHMM_LIBRARY)

INCLUDE_DIRECTORIES(include) #Find the Headers

add_library(CHMM src/HMM.cpp) #Adds a library of for the HMM class

target_include_directories(CHMM
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

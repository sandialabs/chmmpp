# chmmpp

The chmmpp software supports the analysis of multivariate time series
data to detect patterns using a Hidden Markov Model (HMM).

Many applications involve the detection and characterization of hidden or
latent states in a complex system, using observable states and variables.
The chmmpp software supports inference of latent states integrating
both (1) a HMM and (2) application-specific constraints that reflect
known relationships amongst hidden states. For example, HMMs have been
widely used in natural language processing to tag the part of speech
of words in a sentence (e.g. noun, verb, adjective, etc.). But in many
applications there are known relationships that need to be enforced,
such as the fact that a simple English sentence must contain at least
one noun and exactly one verb.

The chmmpp software supports both application-specific and generic methods
for constrained inference.  This includes a framework for customized
Viterbi methods, constrained inference of hidden states with A-star and
integer programming methods, and various contraint-informed methods for
learning HMM model parameters. A focus of chmmpp is support for generic
methods that enable the agile expression of complex sets of constraints
that naturally arise in many real-world applications. Optimization
constraints can be expressed in chmmpp directly in C++ or using the
coek modeling framework. A variety of commercial and open source source
optimization solvers can be used to ensure that maximum likelihood
solutions are found for hidden states.





## Setup

To install the library create a directory called build, navigate to this directory, and run cmake ..

A libary will be created in build/library and executables of the examples will be found in build/examples  


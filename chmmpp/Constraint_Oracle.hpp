#pragma once
#include <vector>

class Constraint_Oracle_Base {
public:
    virtual bool operator()(std::vector<int> hid) = 0;
    bool partial_oracle; // If this is true then we can assess the oracle on partial sequences,
    //e.g. if the oracle return false, then the partial sequence can never be feasible
};
#pragma once
#include <cmath>
#include <string>

namespace chmmpp {

std::pair<bool,std::string> compare(const std::vector<double>& v, const std::vector<double>& b, double tolerance)
{
for (size_t i=0; i<v.size(); i++) {
    if (std::fabs(v[i] - b[i]) > tolerance)
        return {false, "Difference at " + std::to_string(i) + ": " + std::to_string(v[i]) + " and " + std::to_string(b[i])};
    }
return {true,""};
}

std::pair<bool,std::string> compare(const std::vector<std::vector<double>>& v, const std::vector<std::vector<double>>& b, double tolerance)
{
for (size_t i=0; i<v.size(); i++) {
    for (size_t j=0; j<v[i].size(); j++) {
        if (std::fabs(v[i][j] - b[i][j]) > tolerance)
            return {false, "Difference at (" + std::to_string(i) + "," + std::to_string(j) + "): " + std::to_string(v[i][j]) + " and " + std::to_string(b[i][j])};
    }
}
return {true,""};
}

} // namespace chmmpp

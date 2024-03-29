#include <iostream>
#include "Options.hpp"

namespace chmmpp {

    void Options::print_options() const
    {
        std::cout << "Options:" << std::endl;
        for (auto& it : options) {
            std::cout << it.first << ": ";
            std::cout << std::get_if<std::string>(&it.second);
            std::cout << std::get_if<int>(&it.second);
            std::cout << std::get_if<unsigned int>(&it.second);
            std::cout << std::get_if<double>(&it.second);
            std::cout << std::endl;
            }
    }

} // namespace chmmpp

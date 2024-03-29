#include <iostream>
#include "Options.hpp"

namespace chmmpp {

void Options::print_options() const
{
    std::cout << "Options:" << std::endl;
    for (auto& it : options) {
        std::cout << "  " << it.first << ": ";
        if (std::holds_alternative<std::string>(it.second))
            std::cout << std::get<std::string>(it.second);
        else if (std::holds_alternative<int>(it.second))
            std::cout << std::get<int>(it.second);
        else if (std::holds_alternative<unsigned int>(it.second))
            std::cout << std::get<unsigned int>(it.second);
        else if (std::holds_alternative<double>(it.second))
            std::cout << std::get<double>(it.second);
        else
            std::cout << "<empty>";
        std::cout << std::endl;
    }
}

}  // namespace chmmpp

#include <iostream>
#include "Options.hpp"

namespace chmmpp {

// GCOVR_EXCL_START
void Options::print_options() const
{
    std::cout << "Options:" << std::endl;
    for (auto& [key,value] : option_data) {
        std::cout << "  " << key << ": ";
        if (std::holds_alternative<std::string>(value))
            std::cout << std::get<std::string>(value);
        else if (std::holds_alternative<int>(value))
            std::cout << std::get<int>(value);
        else if (std::holds_alternative<unsigned int>(value))
            std::cout << std::get<unsigned int>(value);
        else if (std::holds_alternative<double>(value))
            std::cout << std::get<double>(value);
        else
            std::cout << "<empty>";
        std::cout << std::endl;
    }
}
// GCOVR_EXCL_STOP

}  // namespace chmmpp

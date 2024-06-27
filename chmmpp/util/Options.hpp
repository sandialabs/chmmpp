#pragma once

#include <map>
#include <variant>
#include <string>
#include <iostream>

namespace chmmpp {

//
// A simple class to set options that are used to configure methods in a subclass.
//
class Options {
   public:
    std::map<std::string, std::variant<std::string, int, double, unsigned int>> option_data;

    size_t num_options() { return option_data.size(); }
    void clear_options() { option_data.clear(); }

    void clear_option(const std::string& name)
        { option_data.erase(name); }

    Options& get_options() { return *this; }

    const Options& get_options() const { return *this; }

    template <typename T>
    void set_option(const std::string& name, const T& value)
    {
        option_data[name] = value;
    }

    template <typename T>
    void get_option(const std::string& name, T& value) const
    {
        auto it = option_data.find(name);
        if (it == option_data.end()) return;
        if (std::holds_alternative<T>(it->second)) {
            value = std::get<T>(it->second);
            return;
            }
        std::cerr << "WARNING: unexpected type for option '" << name << "'" << std::endl;
        return;
    }

    void print_options() const;
};

template <>
inline void Options::get_option(const std::string& name, unsigned int& value) const
{
auto it = option_data.find(name);
if (it == option_data.end()) return;

if (std::holds_alternative<unsigned int>(it->second)) {
    value = std::get<unsigned int>(it->second);
    return;
    }
if (std::holds_alternative<int>(it->second)) 
    {
    int tmp = std::get<int>(it->second);
    if (tmp >= 0) {
        value = static_cast<unsigned int>(tmp);
        return;
        }
    }
std::cerr << "WARNING: unexpected type for option '" << name << "'" << std::endl;
return;
}

}  // namespace chmmpp

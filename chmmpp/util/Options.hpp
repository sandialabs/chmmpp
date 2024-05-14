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
    std::map<std::string, std::variant<std::string, int, double, unsigned int>> options;

    void clear_options() { options.clear(); }

    void clear_option(const std::string& name)
        { options.erase(name); }

    Options& get_options() { return *this; }

    const Options& get_options() const { return *this; }

    template <typename T>
    void set_option(const std::string& name, const T& value)
    {
        options[name] = value;
    }

    template <typename T>
    void get_option(const std::string& name, T& value) const
    {
        auto it = options.find(name);
        if (it == options.end()) return;
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
auto it = options.find(name);
if (it == options.end()) return;

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

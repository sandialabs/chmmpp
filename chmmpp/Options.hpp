#pragma once

#include <map>
#include <variant>
#include <optional>
#include <string>

//
// A simple class to set options that are used to configure methods in a subclass.
//
class Options
{
public:

    std::map<std::string, std::variant<std::string, int, double, unsigned int>> options;

    Options& get_options()
        {return *this;}

    const Options& get_options() const
        {return *this;}

    template <typename T>
    void set_option(const std::string& name, const T& value)
        { options[name] = value; }

    template <typename T>
    std::optional<T> get_option(const std::string& name)
        {
        auto it = options.find(name);
        if (it == options.end())
            return std::optional<T>();
        return std::optional<T>(it->second);
        }
};

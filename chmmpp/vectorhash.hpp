#pragma once

#include <vector>
#include <boost/functional/hash.hpp>  //Hash for pairs

namespace chmmpp {

// Hash for vectors
// Taken for Stack Overflow
// (https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/50978188#50978188)
template <typename T>
T xorshift(const T &n, int i)
{
    return n ^ (n >> i);
}

uint64_t distribute(const uint64_t &n);

template <typename T, typename S>
typename std::enable_if<std::is_unsigned<T>::value, T>::type constexpr rotl(const T n, const S i)
{
    const T m = (std::numeric_limits<T>::digits - 1);
    const T c = i & m;
    return (n << c)
           | (n >> ((T(0) - c) & m));  // this is usually recognized by the compiler to mean
                                       // rotation, also c++20 now gives us rotl directly
}

// call this function with the old seed and the new key to be hashed and combined into the new seed
// value, respectively the final hash
template <class T>
inline size_t hash_combine(std::size_t &seed, const T &v)
{
    return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute(std::hash<T>{}(v));
}

template <typename T>
struct vectorHash {
    std::size_t operator()(const std::vector<T> &vec) const
    {
        std::size_t seed = vec.size();
        for (auto &i : vec) {
            seed = hash_combine(seed, i);
        }
        return seed;
    }
};

}  // namespace chmmpp

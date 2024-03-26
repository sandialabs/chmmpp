#include "vectorhash.hpp"

namespace chmmpp {

// a hash function with another name as to not confuse with std::hash
uint64_t distribute(const uint64_t &n)
{
    uint64_t p = 0x5555555555555555ull;    // pattern of alternating 0 and 1
    uint64_t c = 17316035218449499591ull;  // random uneven integer constant;
    return c * xorshift(p * xorshift(n, 32), 32);
}

}  // namespace chmmpp

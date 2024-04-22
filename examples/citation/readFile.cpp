
#include <iostream>
#include <fstream>
#include <sstream>
#include <chmmpp/util/vectorhash.hpp>
#include "citationHMM.hpp"

namespace chmmpp {

// Reads in data from file
// Fle is of the form:
// Word Word ... Word                (counter % 4 == 0)
// Category Category ... Category    (counter % 4 == 1)
//                                   (counter % 4 == 2)
//                                   (coutner % 4 == 3)
void readFile(std::ifstream &inputFile, std::vector<std::vector<std::string> > &words,
              std::vector<std::vector<std::string> > &categories)
{
    std::string line;
    int counter = 0;
    std::vector<std::string> lineVec;

    while (std::getline(inputFile, line)) {
        if (((counter % 4) != 2) && ((counter % 4) != 3)) {
            lineVec.clear();
            std::istringstream iss(line);
            std::string word;

            while (std::getline(iss, word, '\t')) {
                lineVec.push_back(word);
            }

            if ((counter % 4) == 0) {
                words.push_back(lineVec);
            }
            else {
                categories.push_back(lineVec);
            }
        }
        ++counter;
    }

    return;
}

}  // namespace chmmpp

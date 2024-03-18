//main.cpp

#include "HMM.h"

int main() {
    //bool oracleConstraint(std::vector<int> hid, double numZeros);

    std::vector< std::vector<double> > A{{0.899,0.101},{0.099,0.901}}; //Transition Matrix
    std::vector<double> S =  {0.501,0.499}; //Start probabilities 
    std::vector< std::vector<double> > E{{0.699,0.301},{0.299,0.701}}; //Emission Matrix
    
    int T = 50; //Time Horizon
    int numIt = 10; //Number of runs

    //Initial Guesses
    std::vector< std::vector<double> > AInitial{{0.61,0.39},{0.4,0.6}};
    std::vector<double> SInitial{0.51,0.49};
    std::vector< std::vector<double> > EInitial{{0.91,0.09},{0.1,0.9}};

    HMM toLearn1(AInitial,SInitial,EInitial,0); //0 is the seed of the RNG, can remove and it seeds by time
    HMM toLearn2(AInitial,SInitial,EInitial,0);
    HMM toLearn3(AInitial,SInitial,EInitial,0);
    HMM trueHMM(A,S,E,0);

    //Store the observed and hidden variables as well as the number of zeros
    std::vector< std::vector<int> > obs;   
    std::vector< std::vector<int> > hid;
    std::vector<int> numZeros;
    std::vector< std::function<bool(std::vector<int>)> > constraintOracleVec;

    for(int i = 0; i < numIt; ++i) {
        //std::cout << "Iteration number: " << i << "\n";
        obs.push_back({});
        hid.push_back({});

        trueHMM.run(T,obs[i],hid[i]);

        double numZerosTemp = count(hid[i].begin(), hid[i].end(), 0);
        numZeros.push_back(numZerosTemp);
        constraintOracleVec.push_back([numZerosTemp](std::vector<int> myHid) -> bool { return (numZerosTemp == count(myHid.begin(), myHid.end(), 0));});    
        numZeros.push_back(count(hid[i].begin(), hid[i].end(), 0));
    }

    std::cout << "Learning without constraints.\n";
    toLearn1.learn(obs);
    std::cout << "\nSoft learning with constraints.\n";
    toLearn2.learn(obs,numZeros);
    std::cout << "\nHard learning with constraints.\n";
    toLearn3.learnHard(obs,constraintOracleVec,100);

    std::cout << "Learned parameters without constraints:\n\n";
    toLearn1.print();
    std::cout << "Learned parameters with constraints.\n\n";
    toLearn2.print();
    std::cout << "Hard learned parameters with constraints.\n\n";
    toLearn3.print();
    std::cout << "True parameter values:\n\n";
    trueHMM.print();

    return 0;
}
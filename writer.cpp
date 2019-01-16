#include <fstream>
#include <iostream>
#include <cassert>
#include <unistd.h>
#include <vector>

void setLastState( double state[] );

int main(void) {

    double st[] = {0,1,2,3,4,5,6,7,8,9,10,11,12};
    unsigned int microseconds = 100;

    while (true) {
        usleep(microseconds*10);
        st[0] += 1;
        setLastState(st);
    }
    return 0;
}

void setLastState( double state[] ) {

    // We ran this for ~16h and it had no problems,
    // so I think it's pretty safe to use
    std::vector<double> myVec;
    for (int i = 0; i < 13; ++i) {
        myVec.push_back(state[i]);
    }
    std::ofstream outfile;
    outfile.open("state.dat");
    //for (int i = 0; i < 13; ++i) {
    for (int i = 0; i < 13; ++i) {
        //outfile << state[i] << " ";
        outfile << myVec[i] << " ";
    }
    outfile.close();
    
    /*
    if (m_lastState.size() < 13) { // initialize the first time
        for (int i = 0; i < 13; i++) {
            m_lastState.push_back(0);
        }
    }
    //std::cout << "set last state to: ";
    for (int i = 0; i < 13; i++) {
        m_lastState[i] = state[i];
        //std::cout << m_lastState[i] << " ";
    }
    //std::cout << std::endl;
    */
}

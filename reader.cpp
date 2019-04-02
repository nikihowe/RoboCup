#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <unistd.h>
#include <string>
#include <sstream>
#include <sys/file.h>

std::vector<double> getLastState();

int main(void) {
    std::cout << "hello" << std::endl;

    unsigned int microseconds = 100;
    while (true) {
        usleep(microseconds);
        getLastState();
    }
    return 0;
}

std::vector<double> getLastState() {

    FILE *inFile = fopen("ugh.dat", "r"); // read only 
    int fileNo = fileno(inFile);
    int gotLock;
    std::vector<double> toRet;

    std::cout << "waiting to get lock" << std::endl;
    while(gotLock = flock(fileNo, LOCK_EX) != 0) {}; // wait to get lock
    std::cout << "got lock" << std::endl;
    
    for (int i = 0; i < 13; i++) {
        double *a;

        fscanf(inFile, "%lf", a);

        std::cout << "string: " << a << std::endl;

        //toRet.push_back();
    }
    return toRet;
}
          
/*
std::vector<double> getLastState() {

    std::vector<double> toRet;
    std::ifstream infile("state.dat", std::ios::in);

    int count = 0;
    while (toRet.size() < 13) {
        std::cout << "attempt " << count << ": ";
        toRet.clear();
        assert(toRet.size() < 13);
        if (!infile.is_open()) {
            std::cerr << "Probelm opening file" << std::endl;
            assert(false);
        }
        double num = 0;
        while (infile >> num) {
            toRet.push_back(num);
        }
        std::cout << "size is " << toRet.size() << std::endl;
        std::cout << "count: " << count++ << std::endl;
    }

    for (double num : toRet) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return toRet;
}
*/

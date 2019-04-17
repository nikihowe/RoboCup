#include <iostream>
#include <string>
#include <bits/stdc++.h> 
#include <ctime>

#include <unistd.h>
#include <sys/wait.h>

int getExts(const char *fileName) {
   int pid, status;
   // first we fork the process
   if (pid = fork()) {
       // pid != 0: this is the parent process (i.e. our process)
       waitpid(pid, &status, 0); // wait for the child to exit
   } else {
       /* pid == 0: this is the child process. now let's load the */

       /* exec does not return unless the program couldn't be started. 
          when the child process stops, the waitpid() above will return.
       */
       std::string ex = std::string("./dynpartix64 -f ") + fileName
              + std::string(" -s preferred > out.txt");

       int n = ex.length(); 
       char e[n + 1]; 
       strcpy(e, ex.c_str());

       system(e);
   }
   return status; // this is the parent process again.
}

int main() {

   char *myInput = "eg.txt";

   clock_t start = clock();
   getExts(myInput);
   clock_t end = clock();

   std::cout << "start " << start << " end " << end << std::endl;
   std::cout << "time " << (end - start + CLOCKS_PER_SEC)*1.0/CLOCKS_PER_SEC << std::endl;
   std::cout << CLOCKS_PER_SEC << std::endl;

   return 0;
}

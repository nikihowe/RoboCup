#include "SMDPAgent.h"
#include "WorldModel.h"
#include "tiles2.h"
#include "LoggerDraw.h"
#include <cassert>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <sstream>

// Niki includes
#include <utility>
#include <set>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <fstream>
#include <iterator>

#include <sys/wait.h>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#define RL_MEMORY_SIZE 1048576
#define RL_MAX_NONZERO_TRACES 100000
#define RL_MAX_NUM_TILINGS 6000

// Niki defines
#define DEBUGPRINT      0 // for debugging
#define NUM_ACTIONS     4 // for generalizing
#define NUM_STATE_VARS 19 //  "       "    

/* 
 * NOTE: numK = 5, numT = 4
  int j = 0;
  state[ j++ ] = WB_dist_to_C; // 0: distance to center
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = WB_dist_to_K[ i ]; // 1,2,3
  for ( int i = 0; i < numT; i++ )
    state[ j++ ] = WB_dist_to_T[ i ]; // 4,5,6
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = dist_to_C_K[ i ]; // 7,8,9
  for ( int i = 0; i < numT; i++ )
    state[ j++ ] = dist_to_C_T[ i ]; // 10,11,12
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = nearest_Opp_dist_K[ i ]; // 13,14,15
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = nearest_Opp_ang_K[ i ]; // 16,17,18
*/

namespace keepaway_demo {

class ArgumentationAgent: public virtual SMDPAgent {
protected:
  char weightsFile[256];
  bool bLearning;
  bool bSaveWeights;

  /// Hive mind indicator and file descriptor.
  bool hiveMind;
  int hiveFile;
  
  int epochNum;
  int lastAction;
  // Niki-added
  std::vector<double> lastLocalState;
  std::vector<double> curTable;
  std::vector<double> nextTable;

  // Niki
  long episodeCount;
  
  // Added by Niki
  //double lastState[ MAX_STATE_VARS ];
  enum Situation { Safe, UnderThreat, InDanger };
  enum Value { MK, IT, TK };
  enum Argument { H, O1, O2, O3, F1, F2, F3 };
  enum Label { IN, OUT, UNDEC };
  
  // Pre-computed extensions
  // maps applicable arguments + sit -> pref exts
  // for use when using pre-computed extensions
  std::map<std::pair<std::set<Argument>, Situation>, std::set< std::set<Argument> > > myExts;

  double alpha;
  double gamma;
  double lambda;
  double epsilon;

  double tileWidths[ MAX_STATE_VARS ];
  double Q[ MAX_ACTIONS ];

  double* weights;
  double weightsRaw[ RL_MEMORY_SIZE ];
  double traces[ RL_MEMORY_SIZE ];

  int tiles[ MAX_ACTIONS ][ RL_MAX_NUM_TILINGS ];
  int numTilings;

  double minimumTrace;
  int nonzeroTraces[ RL_MAX_NONZERO_TRACES ];
  int numNonzeroTraces;
  int nonzeroTracesInverse[ RL_MEMORY_SIZE ];

  collision_table *colTab;

  // Load / Save weights from/to disk
  bool loadWeights( char *filename );
  bool saveWeights( char *filename );

  // Value function methods for CMACs
  int selectAction();
  int selectBiasedAction(std::vector<double> potTable); // Niki-added
  void initializeTileWidths( int numK, int numT );
  double computeQ( int a );
  int argmaxQ();
  int biasedArgmaxQ(std::vector<double> potTable); // Niki changed
  void updateWeights( double delta );
  virtual void loadTiles( double state[] );

  // Eligibility trace methods
  void clearTrace( int f );
  void clearExistentTrace( int f, int loc );
  void decayTraces( double decayRate );
  void setTrace( int f, float newTraceValue );
  void increaseMinTrace();
public:

  ArgumentationAgent(
	WorldModel& world_, int numFeatures, 
	int numActions, bool bLearn,
  	double widths[], char *loadWeightsFile,
  	char *saveWeightsFile, bool hiveMind);

  // Support for extra modes and/or analysis.
  double getQ(int action);
  void setEpsilon(double epsilon);
  
  // Niki-written startup code
  template <class T>
  std::set< std::set<T> > getAllSubsets(std::vector<T> s);
  void precomputeAllExtensions();

  // Niki-written for sharing potential info
  double getLastpot();
  void setLastpot(double pot);


  void loadExtensions();
  void loadSingleExt( std::ifstream &in );
  std::set<Argument> getArgs(std::string line);
  Situation getSit(std::string line);

  template <class T>
  void printVec( std::vector<T> a, int size ) {
    for (int i = 0; i < size; i++) {
      std::cout << a[i] << " ";
    }
  std::cout << std::endl;
  }

  // Niki-written reward shaping
  double getPotential(double state[], int action);
  std::set< std::set<Argument> > extractIntegerWords(std::string str);
  std::vector<double> getPotentialOverActions(double state[]);
  std::set<Argument> getApplicableArguments(double state[]);
  bool checkOpen(double state[], int i);
  bool checkFar(double state[], int i);
  std::set< std::pair<Argument, Argument> > setAllAttacks(
          std::set<Argument> &args, Situation sit);
  Situation getSituation(double state[]);
  void simplifyFramework(
          std::set< std::pair<Argument, Argument> > &attacks,
          Situation sit);
  double getRelevantPot(Argument arg, Situation sit);
  Value getValue(Argument arg);
  std::set< std::set<Argument> > getPreferredExtensions(
          std::set<Argument> &args,
          std::set< std::pair<Argument, Argument> > &attacks);
  std::set< std::set<Argument> > getExternalSolverPrefExts(
          std::set<Argument> &args,
          std::set< std::pair<Argument, Argument> > &attacks);
  int externalSolver(const char *fileName);
  std::set< std::set<Argument> > getPreferredExtensionsFast(
          double state[], std::set<Argument> &args);
  std::set<Argument> choosePrefExt(std::set< std::set<Argument> > &prefExts);
  int getActionFromExt(std::set<Argument> &prefExt);
  double getGFromExt(std::set<Argument> &prefExt, Situation sit);
  
  // sets the "lastLocalState" vector to the current state
  /*
  void setLastLocalState( double state[] ) { 
      if (lastLocalState.size() < 13) {
          std::cerr << "------last state not initialized!------" << std::endl;
          assert(false);
      }
      for (int i = 0; i < 13; i++) {
          lastLocalState[i] = state[i];
      }
  }
  */

  void printDVec( std::vector<double> a, int size ) {
      for (int i = 0; i < size; i++) {
          std::cout << a[i] << " ";
      }
      std::cout << std::endl;
  }

  void printDArr( double a[], int size ) {
      for (int i = 0; i < size; i++) {
          std::cout << a[i] << " ";
      }
      std::cout << std::endl;
  }


  // Niki-written Labelling code
  void findLabellings(std::map<Argument, Label> ass,
          std::set< std::pair<Argument, Argument> > &attacks,
          std::set< std::map<Argument, Label> > &allLabellings);

  bool isIllegallyIN(Argument arg, std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);
  bool containsIllegallyIN(std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);

  bool isSuperIllegallyIN(Argument arg, std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);
  bool containsSuperIllegallyIN(std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);

  bool isLegallyIN(Argument arg, std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);

  bool isLegallyOUT(Argument arg, std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);
  bool isIllegallyOUT(Argument arg, std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);

  Argument getSuperIllegallyIN(std::map<Argument, Label> &ass,
          std::set< std::pair<Argument, Argument> > &attacks);

  std::map<Argument, Label> transitionStep(std::map<Argument, Label> ass,
          Argument x, std::set< std::pair<Argument, Argument> > &attacks);

  bool isINSubsetOf(std::map<Argument, Label> ass1, std::map<Argument, Label> ass2);

  // SMDP Sarsa implementation
  int  startEpisode( double state[] );
  int  step( double reward, double state[] );
  void endEpisode( double reward );
  void setParams(int iCutoffEpisodes, int iStopLearningEpisodes);
  void shutDown();

  WorldModel& world;
};

}

// cc starts here
// If all is well, there should be no mention of anything keepaway- or soccer-
// related in this file. 

/**
 * Designed specifically to match the serialization format for collision_table.
 * See collision_table::save and collision_table::restore.
 */
#pragma pack(push, 1)
struct CollisionTableHeader {
  long m;
  int safe;
  long calls;
  long clearhits;
  long collisions;
};
#pragma pack(pop)
#define VERBOSE_HIVE_MIND false

/**
 * Assumes collision table header follows weights.
 * Returns the memory location after the header because that's useful for colTab
 * data array.
 */
long* loadColTabHeader(collision_table* colTab, double* weights) {
  CollisionTableHeader* colTabHeader =
    reinterpret_cast<CollisionTableHeader*>(weights + RL_MEMORY_SIZE);
  // Do each field individually, since they don't all line up exactly for an
  // easy copy.
  colTab->calls = colTabHeader->calls;
  colTab->clearhits = colTabHeader->clearhits;
  colTab->collisions = colTabHeader->collisions;
  colTab->m = colTabHeader->m;
  colTab->safe = colTabHeader->safe;
  if (VERBOSE_HIVE_MIND) {
    cout << "Loaded colTabHeader:" << endl
      << " calls: " << colTab->calls << endl
      << " clearhits: " << colTab->clearhits << endl
      << " collisions: " << colTab->collisions << endl
      << " m: " << colTab->m << endl
      << " safe: " << colTab->safe << endl;
  }
  return reinterpret_cast<long*>(colTabHeader + 1);
}

extern LoggerDraw LogDraw;


using namespace keepaway_demo;
using namespace std;


extern "C" {

SMDPAgent* createAgent(
  WorldModel& world_,
  int numFeatures, int numActions, bool bLearn,
  double widths[], char *loadWeightsFile,
  char *saveWeightsFile, bool hiveMind
) {
  ArgumentationAgent* agent = new ArgumentationAgent(
    world_, numFeatures, numActions, bLearn,
    widths, loadWeightsFile,
    saveWeightsFile, hiveMind);
  return agent;
}

}


namespace keepaway_demo {

ArgumentationAgent::ArgumentationAgent( 
  WorldModel& world_, int numFeatures, int numActions, bool bLearn,
  double widths[], char *loadWeightsFile,
  char *saveWeightsFile, bool hiveMind ):
    SMDPAgent( numFeatures, numActions ), hiveFile(-1), world(world_)
{
  //srand(time(NULL)); (already set in main)
  bLearning = bLearn;

  for ( int i = 0; i < getNumFeatures(); i++ ) {
    tileWidths[ i ] = widths[ i ];
  }

  // Saving weights (including for hive mind) requires learning and a file name.
  this->hiveMind = false;
  if ( bLearning && strlen( saveWeightsFile ) > 0 ) {
    strcpy( weightsFile, saveWeightsFile );
    bSaveWeights = true;
    // Hive mind further requires loading and saving from the same file.
    if (!strcmp(loadWeightsFile, saveWeightsFile)) {
      this->hiveMind = hiveMind;
    }
  }
  else {
    bSaveWeights = false;
  }

  alpha = 0.125;
  gamma = 1.0;
  lambda = 0;
  epsilon = 0.01;
  minimumTrace = 0.01;

  epochNum = 0;
  lastAction = -1;
  //lastLocalState = std::vector<double>(25, -1); // initialize state to nothing
  curTable  = std::vector<double>(NUM_ACTIONS, 0);
  nextTable = std::vector<double>(NUM_ACTIONS, 0);
  std::cout << "I am player " << world_.getPlayerNumber() << std::endl;
  loadExtensions();
  std::cout << "loaded " << myExts.size() << " extensions" << std::endl;
  episodeCount = 0;

  numNonzeroTraces = 0;
  weights = weightsRaw;
  for ( int i = 0; i < RL_MEMORY_SIZE; i++ ) {
    weights[ i ] = 0;
    traces[ i ] = 0;
  }

  srand( (unsigned int) 0 );
  int tmp[ 2 ];
  float tmpf[ 2 ];
  colTab = new collision_table( RL_MEMORY_SIZE, 1 );

  GetTiles( tmp, 1, 1, tmpf, 0 );  // A dummy call to set the hashing table    
  srand( time( NULL ) );

  if ( strlen( loadWeightsFile ) > 0 )
    loadWeights( loadWeightsFile );
  std::cout << "I am player " << world_.getPlayerNumber() << std::endl;
}

double ArgumentationAgent::getQ(int action) {
  if (action < 0 || action > getNumActions()) {
    throw "invalid action";
  }
  return Q[action];
}

void ArgumentationAgent::setEpsilon(double epsilon) {
  this->epsilon = epsilon;
}

void ArgumentationAgent::precomputeAllExtensions() {

    //std::map<std::pair<std::set<Argument>, Situation>, std::set< std::set<Argument> > > myExts;

    // All situations
    std::vector<Situation> sits;
    sits.push_back(Safe);
    sits.push_back(UnderThreat);
    sits.push_back(InDanger);

    // Al args
    std::vector<Argument> args;
    args.push_back(H);
    args.push_back(O1);
    args.push_back(O2);
    args.push_back(O3);
    args.push_back(F1);
    args.push_back(F2);
    args.push_back(F3);

    printVec(sits, sits.size());
    printVec(args, args.size());

    //std::set<Argument> s(v.begin(), v.end());
    std::set< std::set<Argument> > allArgSets = getAllSubsets(args);

    /*
     * idea is: for each set of arguments, for each situaton, calculate and store the preferred extensions
     */

    for (Situation sit : sits) {
        for (auto args : allArgSets) {

            // Recurd the current index
            std::pair<std::set<Argument>, Situation>
                current(args, sit);

            // Set up the arguments
            std::set< std::pair<Argument, Argument> > attacks =
                setAllAttacks(args, sit);

            // Simplify based on situation
            simplifyFramework(attacks, sit);

            // Compute extensions (slow bit)
            std::set< std::set<Argument> > prefExts =
                getPreferredExtensions(args, attacks);

            // Save result
            //myExts.insert({current, prefExts}); 

            std::cout << "args" << std::endl;
            for (auto arg : args) {
                std::cout << arg << " ";
            }
            std::cout << std::endl;
            std::cout << "sit" << std::endl;
            std::cout << sit << std::endl;
            std::cout << "exts" << std::endl;
            for (auto ext : prefExts) {
                for (auto arg : ext) {
                    std::cout << arg << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
                
        }
    }
}

//std::map<std::pair<std::set<Argument>, Situation>, std::set< std::set<Argument> > > myExts;
void ArgumentationAgent::loadExtensions() {
    const char *name = "exts4v3.txt";

    std::ifstream in(name);

    if(!in) {
        std::cout << "Cannot open input file." << std::endl;
        return;
    }

    while (in.is_open()) {
        loadSingleExt(in);
        //std::cout << myExts.size() << std::endl;
    }
    std::cout << "done" << std::endl;
}

void ArgumentationAgent::loadSingleExt( std::ifstream &in ) {

    // We'll return these
    std::set<Argument> args;
    Situation sit;
    std::set< std::set<Argument> > exts;

    // For parsing
    std::string str;

    // Get the arguments
    while (getline(in, str)) { // whitespace and "args"
        //std::cout << "str is " << str << std::endl;
        if (str == "args") {
            break;
        } else if (str == "*") {
            in.close();
            return;
        }
    }
    std::cout << str << std::endl;
    std::getline(in, str); // the arguments
    std::cout << str << std::endl;
    if (!str.empty()) {
        args = getArgs(str);
    }

    // Get the situation
    std::getline(in, str); // "sit"
    std::cout << str << std::endl;
    std::getline(in, str); // the situation
    std::cout << str << std::endl;
    sit = getSit(str);

    // Get the extensions
    std::getline(in, str); // "exts"
    std::cout << str << std::endl;
    // the extensions (many lines)
    while (std::getline(in, str) && !str.empty() && str != "*") {
        std::set<Argument> ext = getArgs(str);
        std::cout << str << std::endl;
        exts.insert(ext);
    }
    std::pair<std::set<Argument>, Situation> current(args, sit);
    myExts.insert({current, exts}); 
    if (str == "*") {
        in.close();
        return;
    }
}

std::set<ArgumentationAgent::Argument> ArgumentationAgent::getArgs(std::string str) {
    std::istringstream iss(str);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

//enum Argument { H, O1, O2, O3, F1, F2, F3 };
    std::set<Argument> args;
    for (auto s : tokens) {
        if (s == "0") {
            args.insert(H);
        } else if (s == "1") {
            args.insert(O1);
        } else if (s == "2") {
            args.insert(O2);
        } else if (s == "3") {
            args.insert(O3);
        } else if (s == "4") {
            args.insert(F1);
        } else if (s == "5") {
            args.insert(F2);
        } else if (s == "6") {
            args.insert(F3);
        } else {
            std::cerr << "whyyy" << std::endl;
            assert(false);
        }
    }
    return args;
}

ArgumentationAgent::Situation ArgumentationAgent::getSit(std::string str) {

    std::istringstream iss(str);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

    Situation sit;
//enum Situation { Safe, UnderThreat, InDanger };
    if (str == "0") {
        sit = Safe;
    } else if (str == "1") {
        sit = UnderThreat;
    } else if (str == "2") {
        sit = InDanger;
    } else {
        std::cerr << "whyyy" << std::endl;
        std::cout << str << std::endl;
        assert(false);
    }
    return sit;
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
    }
    return result;
}

int ArgumentationAgent::externalSolver(const char *fileName) {
   int status = 0;
   // Actually, both are fast enough :)
   std::string ex = std::string("./libs/ArgSemSAT/ArgSemSAT -f ") + fileName
          + std::string(" -p EE-PR -fo apx > outExternalSolverMed.txt");
   //std::string ex = std::string("./libs/dynpartix64 -f ") + fileName
          //+ std::string(" -s preferred > outdyna.txt");
   //./ArgSemSAT -f $inputfile -p EE-PR -fo apx

   int n = ex.length(); 
   char e[n + 1]; 
   strcpy(e, ex.c_str());

   exec(e);
   /*
   int pid, status;
   // first we fork the process
   if ((pid = fork())) {
       // pid != 0: this is the parent process (i.e. our process)
       waitpid(pid, &status, 0); // wait for the child to exit
   } else {
       // pid == 0: this is the child process. now let's load the 

       // exec does not return unless the program couldn't be started. 
          //when the child process stops, the waitpid() above will return.
       //
       //std::string ex = std::string("./libs/dynpartix64 -f ") + fileName
              //+ std::string(" -s preferred > outdyna.txt");
       std::string ex = std::string("./libs/ArgSemSAT/ArgSemSAT -f ") + fileName
              + std::string(" -p EE-PR -fo apx > outdyna.txt");
       //./ArgSemSAT -f $inputfile -p EE-PR -fo apx

       int n = ex.length(); 
       char e[n + 1]; 
       strcpy(e, ex.c_str());

       exec(e);
       //system(e);
       //
   }
   */
   return status; // this is the parent process again.
}


std::set< std::set<ArgumentationAgent::Argument> > ArgumentationAgent::getExternalSolverPrefExts(
        std::set<Argument> &args,
        std::set< std::pair<Argument, Argument> > &attacks) {

    clock_t start = clock();
    // Write the framework to file
    ofstream myfile;
    const char *myInput = "inExternalSolverMed.txt";
    myfile.open(myInput);

    //myfile << "\%arguments\n";
    for (Argument arg : args) {
        myfile << "arg(a" << arg << ").\n";
    }

    //myfile << "\%attacks\n";
    for (auto attack : attacks) {
        myfile << "att(a" << attack.first << ",a"
               << attack.second << ").\n";
    }
    myfile.close();
    clock_t end = clock();
    std::cout << "started writing to file at " << start << " and ended at " << end << std::endl;
    std::cout << "writing to file took " << (end - start) * 1.0 / CLOCKS_PER_SEC << std::endl;

    start = clock();
    // Call Dyna
    externalSolver(myInput);
    end = clock();
    std::cout << "started externalSolver at " << start << " and ended at " << end << std::endl;
    std::cout << "calling externalSolver took " << (end - start) * 1.0 / CLOCKS_PER_SEC << std::endl;

    start = clock();
    // Read the output
    std::string data;
    ifstream infile; 
    infile.open("outExternalSolverMed.txt"); 
     
    std::cout << "Reading from the file" << std::endl; 
    infile >> data; // read the file
    std::cout << "The file contains" << std::endl;
    std::cout << data << std::endl;
    infile.close();

    auto prefExts = extractIntegerWords(data);

    std::cout << "extracted" << std::endl;
    std::cout << prefExts.size() << std::endl;

    std::cout << data << std::endl;
    end = clock();
    std::cout << "started reading result at " << start << " and ended at " << end << std::endl;
    std::cout << "reading result took " << (end - start) * 1.0 / CLOCKS_PER_SEC << std::endl;

    return prefExts;
}

std::set< std::set<ArgumentationAgent::Argument> > ArgumentationAgent::extractIntegerWords(std::string str) { 

    std::set< std::set<Argument> > theExts;
    std::set<Argument> theArgs;

    size_t i = 0;
    while (i++ < str.size()) {
        char c = str[i];
        if (c == 'a') {
            size_t intStart = i + 1;
            size_t intEnd1 = str.find(']', i);
            size_t intEnd2 = str.find(',', i);
            size_t intEnd = std::min(intEnd1, intEnd2);
            std::string myStr = str.substr(intStart, intEnd - intStart);
            int myInt = std::stoi(myStr);
            Argument myArg = static_cast<Argument>(myInt);
            theArgs.insert(myArg);
            i = intEnd - 1;
        } else if (c == ']') { // it's the end of the preferred extension
            if (i == str.size()) { // it's the end of the file (== because ++)
                continue;
            } else { // it's just the end of the extension
                theExts.insert(theArgs);
                theArgs.clear();
            }
        }
    }
    return theExts;
}

std::vector<double> ArgumentationAgent::getPotentialOverActions(double state[]) {

    std::vector<double> shaping(NUM_ACTIONS, 0);

    std::set<Argument> args = getApplicableArguments(state);
    Situation sit = getSituation(state);
    
    // Approach 1: below is the problem-specific, fast way of doing it
    //std::set< std::set<Argument> > prefExts = getPreferredExtensionsFast(state, args);

    // XXX
    // Approach 2: this is the slower, but generalizable, way of doing it
    // For now, everything supporting different actions
    // attacks everything else
    std::set< std::pair<Argument, Argument> > attacks =
        setAllAttacks(args, sit);

    // Simplify the framework based on the situation
    simplifyFramework(attacks, sit);
    
    // Approach 2.1
    // Call the external solver
    //clock_t start = clock();
    //std::set< std::set<Argument> > prefExts = 
        //getExternalSolverPrefExts(args, attacks);

    // Approach 2.2
    // Get the preferred extension from the simplified framework
    // NOTE: could use grounded extension in the future
    //std::set< std::set<Argument> > prefExts =
        //getPreferredExtensions(args, attacks);

    // Approach 3: use the pre-computed values
    //std::map<std::pair<std::set<Argument>, Situation>, std::set< std::set<Argument> > > myExts;

    std::pair<std::set<Argument>, Situation> current(args, sit);
    std::set< std::set<Argument> > prefExts = myExts[current];

    //clock_t end = clock();
    //std::cout << "time to retreive " << (end - start)*1.0/CLOCKS_PER_SEC << std::endl;

    // below is single recommended action
    std::set<Argument> ext = choosePrefExt(prefExts);
    int supportedAction = getActionFromExt(ext);

    for (int action = 0; action < NUM_ACTIONS; action++) {
        if (action == supportedAction) {
            shaping[action] += getGFromExt(ext, sit);
            // ^^ multiply by a factor of how much bigger the field is
        }
    }
    return shaping; 
}

// this works
template <class T>
std::set< std::set<T> > ArgumentationAgent::getAllSubsets(std::vector<T> set)
{
    std::vector< std::vector<T> > subset;
    std::vector<T> empty;
    subset.push_back( empty );

    for (size_t i = 0; i < set.size(); i++)
    {
        std::vector< std::vector<T> > subsetTemp = subset;

        for (size_t j = 0; j < subsetTemp.size(); j++)
            subsetTemp[j].push_back( set[i] );

        for (size_t j = 0; j < subsetTemp.size(); j++)
            subset.push_back( subsetTemp[j] );
    }

    std::set< std::set<T> > toReturn;
    for (size_t i = 0; i < subset.size(); i++) {
        std::set<T> miniToReturn;
        for (size_t j = 0; j < subset[i].size(); j++) {
            miniToReturn.insert(subset[i][j]);
        }
        toReturn.insert(miniToReturn);
    }
    return toReturn;
    //return subset;
}

double ArgumentationAgent::getPotential(double state[], int action) {

    //std::cout << "state: ";
    //for (int i = 0; i < 13; i++) {
        //std::cout << state[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << "action: " << action << std::endl;

    //double start = clock();
    // args will contain all applicable arguments
    std::set<Argument> args = getApplicableArguments(state);
    
    // Get the current situation
    Situation sit = getSituation(state);

    /* THIS IS THE STANDARD WAY OF DOING IT, BUT TOO SLOW
    // For now, everything supporting different actions
    // attacks everything else
    std::set< std::pair<Argument, Argument> > attacks =
        setAllAttacks(args, sit);
    //double setAllAttacks = clock();

    // Simplify the framework based on the situation
    simplifyFramework(attacks, sit);
    //double afterSimplified = clock();

    // Get the preferred extension from the simplified framework
    // TODO: still need to implement this. will use labelling system.
    // NOTE: could use grounded extension in the future
    std::set< std::set<Argument> > prefExts =
        getPreferredExtensions(args, attacks);
    */

    // New way of doing extensions (decision tree)
    std::set< std::set<Argument> > prefExts = getPreferredExtensionsFast(state, args);
    //double afterPreferredExts = clock();

    // New way of supporting (support all; not random)
    /*
    double toRet = 0;
    for (auto prefExt : prefExts) {
        int supAct = getActionFromExt(prefExt);
        if (action == supAct) {
            toRet += getGFromExt(prefExt, sit);
        }
    }
    */

    // Randomly choose one of the preferred extensions
    // and get the corresponding action
    std::set<Argument> ext = choosePrefExt(prefExts);
    int supportedAction = getActionFromExt(ext);
    //double afterActionChosen = clock();

    // If the supported action matches the actual action,
    // return the corresponding potential. Else, return 0.
    //std::cout << "chosen action: " << action << std::endl;
    //std::cout << "recommended action: " << supportedAction << std::endl;
    double toRet = 0;
    if (action == supportedAction) {
        //std::cout << "match" << std::endl;
        toRet += getGFromExt(ext, sit);
    } else {
        //std::cout << "mismatch, pot: 0" << std::endl;
    }
    //std::cout << "set attacks " << (setAllAttacks - start)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "simplify " << (afterSimplified - setAllAttacks)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "get preferred " << (afterPreferredExts - afterSimplified)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "get preferred " << (afterPreferredExts - start)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "chose action " << (afterActionChosen - afterPreferredExts)/CLOCKS_PER_SEC << std::endl;
    return toRet;
}

std::set<ArgumentationAgent::Argument> ArgumentationAgent::getApplicableArguments(double state[]) {
    std::set<Argument> args;
    //enum Argument { H, O1, O2, O3, F1, F2, F3 };
    args.insert(H); // H is always supported

    if (checkOpen(state, 1)) {
        args.insert(O1);
    }
    if (checkOpen(state, 2)) {
        args.insert(O2);
    }
    if (checkOpen(state, 3)) {
        args.insert(O3);
    }
    if (checkFar(state, 1)) {
        args.insert(F1);
    }
    if (checkFar(state, 2)) {
        args.insert(F2);
    }
    if (checkFar(state, 3)) {
        args.insert(F3);
    }
    return args;
}

std::set< std::pair<ArgumentationAgent::Argument, ArgumentationAgent::Argument> > ArgumentationAgent::setAllAttacks(
        std::set<ArgumentationAgent::Argument> &args, ArgumentationAgent::Situation sit) {

    std::set< std::pair<Argument, Argument> > attacks;
    if (args.count(H)) { // here, count==contains
        for (Argument arg : args) {
            if (arg != H) {
                attacks.insert(std::make_pair(H, arg));
            }
        }
    }
    if (args.count(O1)) {
        for (Argument arg : args) {
            if (arg != O1 && arg != F1) {
                attacks.insert(std::make_pair(O1, arg));
            }
        }
    }
    if (args.count(O2)) {
        for (auto arg : args) {
            if (arg != O2 && arg != F2) {
                attacks.insert(std::make_pair(O2, arg));
            }
        }
    }
    if (args.count(O3)) {
        for (auto arg : args) {
            if (arg != O3 && arg != F3) {
                attacks.insert(std::make_pair(O3, arg));
            }
        }
    }
    if (args.count(F1)) {
        for (auto arg : args) {
            if (arg != F1 && arg != O1) {
                attacks.insert(std::make_pair(F1, arg));
            }
        }
    }
    if (args.count(F2)) {
        for (auto arg : args) {
            if (arg != F2 && arg != O2) {
                attacks.insert(std::make_pair(F2, arg));
            }
        }
    }
    if (args.count(F3)) {
        for (auto arg : args) {
            if (arg != F3 && arg != O3) {
                attacks.insert(std::make_pair(F3, arg));
            }
        }
    }
    // Print out the simplified attacks
    //for (auto attack : attacks) {
        //std::cout << attack.first << "->" << attack.second << std::endl;
    //}
    return attacks;
}

ArgumentationAgent::Situation ArgumentationAgent::getSituation(
        double state[]) {
    double minDist = state[4]; // the first one must be the closest

    if (minDist > 10) {
        return Safe;
    } else if (minDist > 5) {
        return UnderThreat;
    } else if (minDist >= 0) {
        return InDanger;
    } else {
        std::cout << "------negative distance------" << std::endl;
        assert(false);
        return InDanger;
    }
}

void ArgumentationAgent::simplifyFramework(
        std::set< std::pair<ArgumentationAgent::Argument, ArgumentationAgent::Argument> > &attacks,
        ArgumentationAgent::Situation sit) {

    //std::cout << "situation " << sit << std::endl;
    //std::cout << "starting framework" << std::endl;
    //for (auto attack : attacks) {
        //std::cout << attack.first << "->" << attack.second << std::endl;
    //}
    for (auto attack : attacks) {
        if (getRelevantPot(attack.first, sit) <
                getRelevantPot(attack.second, sit)) {
            attacks.erase(attack);
            //std::cout << getRelevantPot(attack.first, sit) << " < " <<
                //getRelevantPot(attack.second, sit) << std::endl;
        }
    }
    //std::cout << "simplified framework" << std::endl;
    //for (auto attack : attacks) {
        //std::cout << attack.first << "->" << attack.second << std::endl;
    //}
}

ArgumentationAgent::Value ArgumentationAgent::getValue(ArgumentationAgent::Argument arg) {
    if (arg == H) {
        return MK;
    } else if (arg == O1 || arg == O2 || arg == O3) {
        return IT;
    } else if (arg == F1 || arg == F2 || arg == F3) {
        return TK;
    } else {
        assert(false);
        std::cerr << "wrong argument" << std::endl;
        return MK;
    }
}

double ArgumentationAgent::getRelevantPot(
        ArgumentationAgent::Argument arg, ArgumentationAgent::Situation sit) {
    Value val = getValue(arg);
    double toRet;
    switch (sit) {
        case Safe:
            //std::cout << "Safe" << std::endl;
            switch (val) {
                case MK: toRet = 40;
                         break;
                case IT: toRet = 20;
                         break;
                case TK: toRet = 10;
                         break;
            }
            break;
        case UnderThreat:
            //std::cout << "Under Threat" << std::endl;
            switch (val) {
                case MK: toRet = 10;
                         break;
                case IT: toRet = 20;
                         break;
                case TK: toRet = 5;
                         break;
            }
            break;
        case InDanger:
            //std::cout << "In Danger" << std::endl;
            switch (val) {
                case MK: toRet = 0;
                         break;
                case IT: toRet = 25;
                         break;
                case TK: toRet = 5;
                         break;
            }
            break;
        default:
            assert(false);
            std::cerr << "wrong value" << std::endl;
            return -1;
    }
    return toRet;
}

// Get all preferred extensions, with scenario-specific optimization
std::set< std::set<ArgumentationAgent::Argument> > ArgumentationAgent::getPreferredExtensionsFast(double state[], std::set<Argument> &args) {

    Situation sit = getSituation(state);

    std::set< std::set<Argument> > prefExts;

    if (sit == Safe) { // H will win always
        std::set<Argument> ext0 = { H };
        prefExts.insert(ext0);
    } else if (sit == UnderThreat) { // O will win if present, and will carry F
        if (args.count(O1) || args.count(O2)) { // O is present
            // add O1 (and then F1 if present)
            if (args.count(O1)) {
                std::set<Argument> ext1;
                ext1.insert(O1);
                if (args.count(F1)) {
                    ext1.insert(F1);
                }
                prefExts.insert(ext1);
            }
            // add O2 (and then F2 if present)
            if (args.count(O2)) {
                std::set<Argument> ext2;
                ext2.insert(O2);
                if (args.count(F2)) {
                    ext2.insert(F2);
                }
                prefExts.insert(ext2);
            }
        } else { // O not present
            std::set<Argument> ext0 = { H };
            prefExts.insert(ext0);
        }
    } else if (sit == InDanger) {
        // O and F will win if present
        if (args.count(O1) || args.count(O2) || args.count(F1) || args.count(F2)) { // O or F
            if (args.count(O1) || (args.count(F1) && !args.count(O2))) { // O1 or F1 present
                std::set<Argument> ext1;
                if (args.count(O1)) {
                    ext1.insert(O1);
                    if (args.count(F1)) {
                        ext1.insert(F1);
                    }
                } else if (args.count(F1) && !args.count(O2)) { // F1 present
                    ext1.insert(F1);
                }
                prefExts.insert(ext1);
            }
            if (args.count(O2) || (args.count(F2) && !args.count(O1))) { // O2 or F2 present
                std::set<Argument> ext2;
                if (args.count(O2)) {
                    ext2.insert(O2);
                    if (args.count(F2)) {
                        ext2.insert(F2);
                    }
                } else if (args.count(F2) && !args.count(O1)) { // F2 present
                    ext2.insert(F2);
                }
                prefExts.insert(ext2);
            }
        } else { // only H present
            std::set<Argument> ext0 = { H };
            prefExts.insert(ext0);
        } 
    } else {
        // should never get here
        assert(false);
        std::cerr << "incorrect situation" << std::endl;
    }
    return prefExts;
}

// Get all preferred extensions from argumentation framework
std::set< std::set<ArgumentationAgent::Argument> > ArgumentationAgent::getPreferredExtensions(
        std::set<ArgumentationAgent::Argument> &args,
        std::set< std::pair<ArgumentationAgent::Argument, ArgumentationAgent::Argument> > &attacks) {

    //double start = clock();
    std::set< std::map<Argument, Label> > allLabellings;
    std::map<Argument, Label> allIN;// = { {H, IN}, {O1, IN}, {O2, IN}, {F1, IN}, {F2, IN} };
    for (Argument arg : args) {
        allIN[arg] = IN;
    }
    //double allINTime = clock();

    // Fill candidateLabels with labellings
    findLabellings(allIN, attacks, allLabellings);
    //double foundLabs = clock();
    
    // Print everything in the allowed labellings
    //for (auto argLabMap : allLabellings) {
        //for (auto argLab : argLabMap) {
            //std::cout << argLab.first << ":=" << argLab.second << ", ";
        //}
        //std::cout << std::endl;
    //}

    std::set< std::set<Argument> > prefExts;

    for (auto extLabels : allLabellings) {
        std::set<Argument> ext;

        std::map<Argument, Label>::iterator it;
        for (it = extLabels.begin(); it != extLabels.end(); it++) {
            if (it->second == IN) {
                ext.insert(it->first); // put the key in ext
            }
        }
        prefExts.insert(ext);
    }
    //double builtPrefExts = clock();

    //std::cout << "all in " << (allINTime - start) / CLOCKS_PER_SEC << std::endl;
    //std::cout << "found labellings " << (foundLabs - allINTime) / CLOCKS_PER_SEC << std::endl;
    //std::cout << "built pref exts " << (builtPrefExts - foundLabs) / CLOCKS_PER_SEC << std::endl;
    //std::cout << "preferred extensions:" << std::endl;
    //for (auto ext : prefExts) {
        //for (Argument arg : ext) {
            //std::cout << arg << ", ";
        //}
        //std::cout << std::endl;
    //}
    return prefExts;

    /*
    std::set< std::set<Argument> > prefExts;
    std::set<Argument> prefExt;

    std::vector<Argument> argVec = 
    void ArgumentationAgent::findLabellings(

    prefExt.insert(H);
    prefExts.insert(prefExt);
    return prefExts;
    */
}

std::set<ArgumentationAgent::Argument> ArgumentationAgent::choosePrefExt(
        std::set< std::set<Argument> > &prefExts) {
    int n = prefExts.size();
    //std::cout << "how many pref exts: " << n << std::endl;
    int choice = rand() % n;
    //std::cout << "choice: " << choice << std::endl;
    std::set< std::set<Argument> >::iterator iter = prefExts.begin();
    for ( ; choice > 0; choice--) { // cycle through elements until we get the right one
        if (iter == prefExts.end()) {
            assert(false);
            std::cerr << "problematic choice" << std::endl;
        }
        iter++;
    } // now we're pointing to the correct element
    return *iter;
}

int ArgumentationAgent::getActionFromExt(std::set<Argument> &args) {
    // Get any Argument from the preferred extension (all will promote same action)
    std::set<Argument>::iterator iter = args.begin();
    Argument arg = *iter;
    if (arg == H) {
        return 0;
    } else if (arg == O1 || arg == F1) {
        return 1;
    } else if (arg == O2 || arg == F2) {
        return 2;
    } else if (arg == O3 || arg == F3) {
        return 3;
    } else {
        assert(false);
        std::cerr << "wrong argument" << std::endl;
        return -1;
    }
}

double ArgumentationAgent::getGFromExt(std::set<Argument> &args, Situation sit) {
    double total = 0;
    for (auto arg : args) {
        //std::cout << "argument " << arg << ": " << getRelevantPot(arg, sit) << std::endl;
        total += getRelevantPot(arg, sit);
    }
    //double scaling = episodeCount > 1000 ? 0 : 1;
    //max(0, 1.0 - episodeCount * 1.0 / 4000);
    double scaling = pow(0.8, episodeCount / 1000);
    return total * scaling;
}

bool ArgumentationAgent::checkOpen(double state[], int i) {
    int minAngle = 15;
    if (i == 1) {
        return state[16] >= minAngle;
    } else if (i == 2) {
        return state[17] >= minAngle;
    } else if (i == 3) {
        return state[18] >= minAngle;
    } else {
        assert(false);
        std::cout << "------checking wrong keeper------" << std::endl;
        return false;
    }
}

bool ArgumentationAgent::checkFar(double state[], int i) {
    int farDistance = 10;
    if (i == 1) {
        return state[13] >= farDistance;
    } else if (i == 2) {
        return state[14] >= farDistance;
    } else if (i == 3) {
        return state[15] >= farDistance;
    } else {
        assert(false);
        std::cout << "------checking wrong keeper------" << std::endl;
        return false;
    }
}

void ArgumentationAgent::findLabellings(std::map<Argument, Label> ass,
      std::set< std::pair<Argument, Argument> > &attacks,
      std::set< std::map<Argument, Label> > &allLabellings) {
    // line 11 of algo
    //double start = clock();
    for (auto tempAss : allLabellings) {
        // if ass subset of tempAss
        if (isINSubsetOf(ass, tempAss)) {
            return;
        }
    }
    //double subCheck = clock();

    // line 14 of algo
    if (!(containsIllegallyIN(ass, attacks))) {
        std::set< std::map<Argument, Label> >::iterator it;
        for (it = allLabellings.begin(); it != allLabellings.end(); it++) {
            // if *it subset of ass
            if (isINSubsetOf((*it), ass)) {
            //if (std::includes(it->begin(), it->end(), ass.begin(), ass.end())) {
                allLabellings.erase(*it);
            }
        }
        allLabellings.insert(ass);
    } else {
        if (containsSuperIllegallyIN(ass, attacks)) {
            auto x = getSuperIllegallyIN(ass, attacks);
            findLabellings(transitionStep(ass, x, attacks),
                    attacks, allLabellings);
        } else {
            std::map<Argument, Label>::iterator it;
            for (it = ass.begin(); it != ass.end(); it++) {
                Argument arg = it->first;
                if (isIllegallyIN(arg, ass, attacks)) {
                    findLabellings(transitionStep(ass, arg, attacks),
                            attacks, allLabellings);
                }
            }
        }
    }
    //double end = clock();
    //std::cout << "sub check " << (subCheck - start) / CLOCKS_PER_SEC << std::endl;
    //std::cout << "rest " << (end - subCheck) / CLOCKS_PER_SEC << std::endl;
    return;
}

bool ArgumentationAgent::isINSubsetOf(
        std::map<Argument, Label> ass1, std::map<Argument, Label> ass2) {
    for (auto argLab : ass1) {
        Argument arg = argLab.first;
        Label lab = argLab.second;
        if (lab == IN) {
            if (!ass2.count(arg) || ass2[arg] != IN) {
                return false;
            }
        }
    }
    return true;
}

bool ArgumentationAgent::containsIllegallyIN(std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {
    std::map<Argument, Label>::iterator it;
    for (it = ass.begin(); it != ass.end(); it++) {
        Argument arg = it->first;
        if (isIllegallyIN(arg, ass, attacks)) {
            return true;
        }
    }
    return false;
}

bool ArgumentationAgent::isIllegallyIN(Argument arg, std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {
    return ass[arg] == IN && !isLegallyIN(arg, ass, attacks);
}

bool ArgumentationAgent::isLegallyIN(Argument arg, std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {
    if (ass[arg] != IN) { return false; }

    for (auto attack : attacks) {
        if (attack.second == arg) {
            Argument attackingArg = attack.first;
            if(ass[attackingArg] != OUT) {
                return false;
            }
        }
    }
    return true;
}

bool ArgumentationAgent::isSuperIllegallyIN(Argument arg, std::map<Argument, Label> &ass,
      std::set< std::pair<Argument, Argument> > &attacks) {
        
    if (isIllegallyIN(arg, ass, attacks)) {
        for (auto attack : attacks) {
            if (attack.second == arg &&
               (ass[attack.first] == UNDEC ||
                isLegallyIN(attack.first, ass, attacks))) {
                return true;
            }
        }
    }
    return false;
}

bool ArgumentationAgent::containsSuperIllegallyIN(std::map<Argument, Label> &ass,
      std::set< std::pair<Argument, Argument> > &attacks) {

    std::map<Argument, Label>::iterator it;
    for (it = ass.begin(); it != ass.end(); it++) {
        Argument arg = it->first;
        if (isSuperIllegallyIN(arg, ass, attacks)) {
            return true;
        }
    }
    return false;
}

ArgumentationAgent::Argument ArgumentationAgent::getSuperIllegallyIN(
        std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {

    std::map<Argument, Label>::iterator it;
    for (it = ass.begin(); it != ass.end(); it++) {
        Argument arg = it->first;
        if (isSuperIllegallyIN(arg, ass, attacks)) {
            return arg;
        }
    }
    // should never get here
    assert(false);
    std::cerr << "doesn't contain SuperIllegal" << std::endl;
    return H;
}

bool ArgumentationAgent::isIllegallyOUT(Argument arg, std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {
    return ass[arg] == OUT && !isLegallyOUT(arg, ass, attacks);
}

bool ArgumentationAgent::isLegallyOUT(Argument arg, std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {
    if (ass[arg] != OUT) {
        return false;
    }

    for (auto attack : attacks) {
        if (attack.second == arg) {
            if (ass[attack.first] == IN) {
                return true;
            }
        }
    }
    return false;
}

std::map<ArgumentationAgent::Argument, ArgumentationAgent::Label> ArgumentationAgent::transitionStep(std::map<Argument, Label> ass,
        Argument x, std::set< std::pair<Argument, Argument> > &attacks) {
    ass[x] = OUT; // only modify local version

    std::map<Argument, Label>::iterator it;
    for (it = ass.begin(); it != ass.end(); it++) {
        Argument arg = it->first;
        if (isIllegallyOUT(arg, ass, attacks)) {
            ass[arg] = UNDEC;
        }
    }
    return ass;
}

double ArgumentationAgent::getLastpot() {
    std::ifstream infile("potMed.dat", std::ios::in);
    double pot;
    infile >> pot;
    return pot;
}

void ArgumentationAgent::setLastpot( double pot ) {
    std::ofstream outfile;
    outfile.open("potMed.dat");
    outfile << pot;
    outfile.close();
}

// End of Niki-written
int ArgumentationAgent::startEpisode( double state[] )
{
    double start = clock();
    //std::cout << clock() << std::endl;;
    //std::cout << "Episode Start. State: ";
    //for (int i = 0; i < 13; i++) {
        //std::cout << state[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << getSituation(state) << std::endl;

  if (hiveMind) loadColTabHeader(colTab, weights);
  epochNum++;
  decayTraces( 0 );
  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    Q[ a ] = computeQ( a ); // calculate Q for this state, across all actions
  }

  
  if (DEBUGPRINT) {
      std::cout << "player " << world.getPlayerNumber() << " start!" << std::endl;
      //std::cout << "last local action " << lastAction << std::endl;
      //std::cout << "last WM action " << world.getLastAction() << std::endl;
      //std::cout << "time last action " << world.getTimeLastAction() << std::endl;
      //std::cout << "last local state  ";
      //printDVec(lastLocalState);
  }
  // TODO
  //lastAction = selectAction(); TODO: this is what we would usually use
  curTable = getPotentialOverActions(state); // fill the curr table
  //printDVec(curTable, NUM_ACTIONS);
  lastAction = selectBiasedAction(curTable);
  // TODO ^^ choosing a biased action
  if (DEBUGPRINT) {
      //std::cout << "curr state  ";
      //printDArr(state);
      //std::cout << "curr action " << lastAction << std::endl;
  }

  char buffer[128];
  sprintf( buffer, "Q[%d] = %.2f", lastAction, Q[lastAction] );
  LogDraw.logText( "Qmax", VecPosition( 25, -30 ),
                   buffer,
                   1, COLOR_BROWN );

  for ( int j = 0; j < numTilings; j++ )
    setTrace( tiles[ lastAction ][ j ], 1.0 );
  if (hiveMind) saveWeights(weightsFile);

  // These are the only things that have changed
  //setLastLocalState( state );
  //world.setLastGlobalAction( lastAction );
  //world.setLastGlobalState( state );
  if (DEBUGPRINT) {
      std::cout << "curr time " << world.getCurrentTime() << std::endl;
  }
  double end = clock();
  if ((end - start)*1.0/CLOCKS_PER_SEC >= 0.09) { // actually 0.1s
    std::cerr << "too slow" << std::endl;
    std::cerr << "---------------------------------------------------------------" << std::endl;
  } 
  setLastpot(curTable[lastAction]); // set to the pot of this action
  return lastAction;
}


int ArgumentationAgent::step( double reward, double state[] )
{
    //std::cout << clock() << std::endl;;
    //std::cout << "Step. ";
    //std::cout << "Last state: ";
    //std::vector<double> lastState = world.getLastGlobalState();
    //for (int i = 0; i < 13; i++) {
        //std::cout << lastState[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << "Last action: " << world.getLastAction() << std::endl;
    //std::cout << "Current state: ";
    //for (int i = 0; i < 13; i++) {
        //std::cout << state[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << getSituation(state) << std::endl;
  double start = clock();
  // Niki-written (line 9)
  double oldPotential = 0;
  if (bLearning) {
    //std::vector<double> ls = world.getLastGlobalState();
    //assert(ls.size() == 13);
    //double *a = &ls[0];
    //double *a = &lastLocalState[0];
    //oldPotential = getPotential(a, world.getLastGlobalAction());
    // TODO
    //oldPotential = getPotential(a, lastAction);
    //oldPotential = curTable[lastAction];
    oldPotential = getLastpot(); // this is from the file
    // TODO
    //std::cout << "old potential: " << oldPotential << std::endl;
  }
  //std::cout << getSituation(lastState) << " " << lastAction << " -> ";

  if (hiveMind) loadColTabHeader(colTab, weights);
  double delta = reward - Q[ lastAction ];
  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    Q[ a ] = computeQ( a ); // calculate Q for this state, across all actions
  }

  if (DEBUGPRINT) {
      std::cout << "player " << world.getPlayerNumber() << " step" << std::endl;
      //std::cout << "last local state  ";
      //printDVec(lastLocalState);
      //std::cout << "last local action " << lastAction << std::endl;
      //std::cout << "last WM action " << world.getLastAction() << std::endl;
      //std::cout << "time last action " << world.getTimeLastAction() << std::endl;
      std::cout << "reward ---       " << reward << std::endl;
  }
  //lastAction = selectAction();
  // TODO
  nextTable = getPotentialOverActions(state); // fill the curr table
  //printDVec(nextTable, NUM_ACTIONS);
  lastAction = selectBiasedAction(nextTable);
  // TODO
  if (DEBUGPRINT) {
      //std::cout << "curr state  ";
      //printDArr(state);
      //std::cout << "curr action " << lastAction << std::endl;
  }

  char buffer[128];
  sprintf( buffer, "Q[%d] = %.2f", lastAction, Q[lastAction] );
  LogDraw.logText( "Qmax", VecPosition( 25, -30 ),
                   buffer,
                   1, COLOR_BROWN );

  if ( !bLearning )
    return lastAction;

  //char buffer[128];
  sprintf( buffer, "reward: %.2f", reward ); 
  LogDraw.logText( "reward", VecPosition( 25, 30 ),
                   buffer,
                   1, COLOR_NAVY );

  delta += Q[ lastAction ]; // Assumes gamma==1
  // ^ this is Q(s', a') because we recalculated in 955

  // Niki-written
  //TODO
  //double newPotential = getPotential(state, lastAction);
  double newPotential = nextTable[lastAction];
  //TODO
  //std::cout << "new potential: " << newPotential << std::endl;
  delta += newPotential - oldPotential; // Assumes gamma==1
  //std::cout << getSituation(state) << " " << lastAction << std::endl;
  //std::cout << newPotential << " - " << oldPotential << " = ";
  //std::cout << newPotential - oldPotential << std::endl << std::flush;

  //std::cout << "Reward before shaping: " << reward << std::endl;
  //std::cout << "Reward after shaping:  " << reward + newPotential - oldPotential << "\n" << std::endl;

  updateWeights( delta );
  Q[ lastAction ] = computeQ( lastAction ); // need to redo because weights changed
  decayTraces( gamma * lambda );

  for ( int a = 0; a < getNumActions(); a++ ) {  //clear other than F[a]
    if ( a != lastAction ) {
      for ( int j = 0; j < numTilings; j++ )
        clearTrace( tiles[ a ][ j ] );
    }
  }
  for ( int j = 0; j < numTilings; j++ )      //replace/set traces F[a]
    setTrace( tiles[ lastAction ][ j ], 1.0 );

  if (hiveMind) saveWeights(weightsFile);

  // Niki-written
  //for (int i = 0; i < MAX_STATE_VARS; i++) {
      //lastState[i] = state[i];
  //}
  // This is actually handled automatically by the world model
  double end = clock();

  //setLastLocalState( state );
  //world.setLastGlobalAction( lastAction );
  //world.setLastGlobalState( state );
  
  if ((end - start)*1.0/CLOCKS_PER_SEC >= 0.09) { // actually 0.1s
    std::cerr << "too slow" << std::endl;
    std::cerr << "---------------------------------------------------------------" << std::endl;
  } 
  if (DEBUGPRINT) {
      std::cout << "shaping          " << newPotential - oldPotential << std::endl;
      std::cout << "curr time " << world.getCurrentTime() << std::endl;
  }
  //TODO
  curTable = nextTable; 
  //TODO
  setLastpot(newPotential); // update the potential
  return lastAction;
}


void ArgumentationAgent::endEpisode( double reward )
{
  double start = clock();
  if (hiveMind) loadColTabHeader(colTab, weights);
  if ( bLearning && lastAction != -1 ) { /* otherwise we never ran on this episode */
    char buffer[128];
    sprintf( buffer, "reward: %.2f", reward ); 
    LogDraw.logText( "reward", VecPosition( 25, 30 ),
                     buffer,
                     1, COLOR_NAVY );

    /* finishing up the last episode */
    /* assuming gamma = 1  -- if not,error*/
    if ( gamma != 1.0)
      cerr << "We're assuming gamma's 1" << endl;
    double delta = reward - Q[ lastAction ];

    //std::cout << clock() << std::endl;;
    //std::cout << "End of Episode" << std::endl;
    //std::vector<double> lastState = world.getLastGlobalState();
    //std::cout << "Last state: ";
    //std::vector<double> ls = world.getLastGlobalState();
    //for (int i = 0; i < 13; i++) {
        //std::cout << ls[i] << " ";
    //}
    //std::cout << "Last action: " << world.getLastGlobalAction() << std::endl;
    //std::cout << std::endl;
    // Niki-written
    //std::vector<double> ls = world.getLastGlobalState();
    //double *a = &ls[0];
    //double *a = &lastLocalState[0];
    if (DEBUGPRINT) {
      std::cout << "player " << world.getPlayerNumber() << " stop!" << std::endl;
      //std::cout << "last local state  ";
      //printDVec(lastLocalState);
      //std::cout << "last local action " << lastAction << std::endl;
      //std::cout << "last WM action " << world.getLastAction() << std::endl;
      //std::cout << "time last action " << world.getTimeLastAction() << std::endl;
      std::cout << "reward ---       " << reward << std::endl;
      std::cout << "curr time " << world.getCurrentTime() << std::endl;
    }
    //double oldPotential = getPotential(a, world.getLastGlobalAction());
    // TODO
    //double oldPotential = getPotential(a, lastAction); // 
    //double oldPotential = curTable[lastAction];
    double oldPotential = getLastpot(); // from the file
    // TODO
    //std::cout << "old potential: " << std::endl;
    delta -= oldPotential; // newPotential==0
    if (DEBUGPRINT) {
        std::cout << "shaping          " << (-1)*oldPotential << std::endl;
    }
    //std::cout << "Reward before shaping: " << reward << std::endl;
    //std::cout << "Reward after shaping:  " << reward - oldPotential << std::endl;
    //std::cout << "****End Episode****" << std::endl;
    updateWeights( delta );
    // TODO Actually, there's still possibly risk for trouble here with multiple
    // TODO players stomping each other. Is this okay?
    // TODO The weight updates themselves are in order.
  }
  // Save weights at random intervals
  if ( bLearning && bSaveWeights && rand() % 200 == 0 && !hiveMind ) {
    saveWeights( weightsFile );
  }
  if (hiveMind) saveWeights(weightsFile);
  lastAction = -1;
  double end = clock();
  if ((end - start)*1.0/CLOCKS_PER_SEC >= 0.09) { // actually 0.1s
    std::cerr << "too slow" << std::endl;
    std::cerr << "---------------------------------------------------------------" << std::endl;
  } 
  episodeCount++;
}

void ArgumentationAgent::shutDown()
{
  // We usually only save weights at random intervals.
  // Always save at shutdown (if we are in saving mode).
  if ( bLearning && bSaveWeights ) {
    cout << "Saving weights at shutdown." << endl;
    saveWeights( weightsFile );
  }
  // Also shut down the hive mind if needed.
  if (hiveMind) {
    size_t mapLength =
      RL_MEMORY_SIZE * sizeof(double) +
      sizeof(CollisionTableHeader) +
      colTab->m * sizeof(long);
    munmap(weights, mapLength);
    close(hiveFile);
    hiveFile = -1;
    // Go back to the own arrays, since our map is no longer valid.
    weights = weightsRaw;
    colTab->data = new long[colTab->m];
  }
}

int ArgumentationAgent::selectAction()
{
  int action;

  // Epsilon-greedy
  if ( bLearning && drand48() < epsilon ) {     /* explore */
    action = rand() % getNumActions();
  }
  else{
    action = argmaxQ();
  }

  return action;
}

int ArgumentationAgent::selectBiasedAction(std::vector<double> potTable) {
  int action;

  // Epsilon-greedy
  if ( bLearning && drand48() < epsilon ) {     /* explore */
    action = rand() % getNumActions();
  }
  else{
    action = biasedArgmaxQ(potTable); // Niki changed
  }

  return action;
}

bool ArgumentationAgent::loadWeights( char *filename )
{
  cout << "Loading weights from " << filename << endl;
  if (hiveMind) {
    if (hiveFile < 0) {
      // First, check the lock file, so we have only one initializer.
      // Later interaction should be approximately synchronized by having only
      // one active player at a time per team, but we can't assume that here.
      stringstream lockNameBuffer;
      lockNameBuffer << filename << ".lock";
      const char* lockName = lockNameBuffer.str().c_str();
      int lock;
      // 10ms delay (times a million to convert from nanos).
      timespec sleepTime = {0, 10 * 1000 * 1000};
      while (true) {
        lock = open(lockName, O_CREAT | O_EXCL, 0664);
        if (lock >= 0) break;
        nanosleep(&sleepTime, NULL);
      }
      // First, see if the file is already there.
      bool fileFound = !access(filename, F_OK);
      // TODO Extract constant for permissions (0664)?
      hiveFile = open(filename, O_RDWR | O_CREAT, 0664);
      size_t mapLength =
        RL_MEMORY_SIZE * sizeof(double) +
        sizeof(CollisionTableHeader) +
        colTab->m * sizeof(long);
      if (!fileFound) {
        // Make the file the right size.
        cout << "Initializing new hive file." << endl;
        if (lseek(hiveFile, mapLength - 1, SEEK_SET) < 0) {
          throw "failed to seek initial file size";
        }
        if (write(hiveFile, "", 1) < 0) {
          throw "failed to expand initial file";
        }
      }
      if (hiveFile < 0) throw "failed to open hive file";
      void* hiveMap =
        mmap(NULL, mapLength, PROT_READ | PROT_WRITE, MAP_SHARED, hiveFile, 0);
      if (hiveMap == MAP_FAILED) throw "failed to map hive file";
      // First the weights.
      weights = reinterpret_cast<double*>(hiveMap);
      // Now the collision table header.
      CollisionTableHeader* colTabHeader =
        reinterpret_cast<CollisionTableHeader*>(weights + RL_MEMORY_SIZE);
      if (fileFound) {
        loadColTabHeader(colTab, weights);
      }
      // Now the collision table data.
      delete[] colTab->data;
      colTab->data = reinterpret_cast<long*>(colTabHeader + 1);
      if (!fileFound) {
        // Clear out initial contents.
        // The whole team might be doing this at the same time. Is that okay?
        for ( int i = 0; i < RL_MEMORY_SIZE; i++ ) {
          weights[ i ] = 0;
        }
        colTab->reset();
        // Make sure the header goes out to the file.
        saveWeights(weightsFile);
      }
      // TODO Separate file lock type with destructor?
      unlink(lockName);
    }
  } else {
    int file = open( filename, O_RDONLY );
    read( file, (char *) weights, RL_MEMORY_SIZE * sizeof(double) );
    colTab->restore( file );
    close( file );
  }
  cout << "...done" << endl;
  return true;
}

bool ArgumentationAgent::saveWeights( char *filename )
{
  if (hiveMind) {
    // The big arrays should be saved out automatically, but we still need to
    // handle the collision table header.
    CollisionTableHeader* colTabHeader =
      reinterpret_cast<CollisionTableHeader*>(weights + RL_MEMORY_SIZE);
    // Do each field individually, since they don't all line up exactly for an
    // easy copy.
    colTabHeader->calls = colTab->calls;
    colTabHeader->clearhits = colTab->clearhits;
    colTabHeader->collisions = colTab->collisions;
    colTabHeader->m = colTab->m;
    colTabHeader->safe = colTab->safe;
    if (VERBOSE_HIVE_MIND) {
      cout << "Saved colTabHeader:" << endl
        << " calls: " << colTab->calls << endl
        << " clearhits: " << colTab->clearhits << endl
        << " collisions: " << colTab->collisions << endl
        << " m: " << colTab->m << endl
        << " safe: " << colTab->safe << endl;
    }
  } else {
    int file = open( filename, O_CREAT | O_WRONLY, 0664 );
    write( file, (char *) weights, RL_MEMORY_SIZE * sizeof(double) );
    colTab->save( file );
    close( file );
  }
  return true;
}

// Compute an action value from current F and theta    
double ArgumentationAgent::computeQ( int a )
{
  double q = 0;
  for ( int j = 0; j < numTilings; j++ ) {
    q += weights[ tiles[ a ][ j ] ];
  }

  return q;
}

// Returns index (action) of largest entry in Q array, breaking ties randomly 
int ArgumentationAgent::argmaxQ()
{
  int bestAction = 0;
  double bestValue = Q[ bestAction ];
  int numTies = 0;
  for ( int a = bestAction + 1; a < getNumActions(); a++ ) {
    double value = Q[ a ];
    if ( value > bestValue ) {
      bestValue = value;
      bestAction = a;
    }
    else if ( value == bestValue ) { // choose randomly for tie
      numTies++;
      if ( rand() % ( numTies + 1 ) == 0 ) {
        bestValue = value;
        bestAction = a;
      }
    }
  }

  return bestAction;
}

// Calculate the best next action, taking into consideration
// the potential value based on current state
int ArgumentationAgent::biasedArgmaxQ(std::vector<double> potTable) {
  int bestAction = 0;
  double bestValue = Q[ bestAction ] + potTable[ bestAction ]; // N added pot
  int numTies = 0;
  for ( int a = bestAction + 1; a < getNumActions(); a++ ) {
    double value = Q[ a ] + potTable[ a ]; // N added pot
    if ( value > bestValue ) {
      bestValue = value;
      bestAction = a;
    }
    else if ( value == bestValue ) { // choose randomly for tie
      numTies++;
      if ( rand() % ( numTies + 1 ) == 0 ) {
        bestValue = value;
        bestAction = a;
      }
    }
  }

  return bestAction;
}

void ArgumentationAgent::updateWeights( double delta )
{
  double tmp = delta * alpha / numTilings;
  for ( int i = 0; i < numNonzeroTraces; i++ ) {
    int f = nonzeroTraces[ i ];
    if ( f > RL_MEMORY_SIZE || f < 0 )
      cerr << "f is too big or too small!!" << f << endl;
    weights[ f ] += tmp * traces[ f ];
    //cout << "weights[" << f << "] = " << weights[f] << endl;
  }
}

void ArgumentationAgent::loadTiles( double state[] )
{
  int tilingsPerGroup = 32;  /* num tilings per tiling group */
  numTilings = 0;

  /* These are the 'tiling groups'  --  play here with representations */
  /* One tiling for each state variable */
  for ( int v = 0; v < getNumFeatures(); v++ ) {
    for ( int a = 0; a < getNumActions(); a++ ) {
      GetTiles1( &(tiles[ a ][ numTilings ]), tilingsPerGroup, colTab,
                 state[ v ] / tileWidths[ v ], a , v );
    }  
    numTilings += tilingsPerGroup;
  }
  if ( numTilings > RL_MAX_NUM_TILINGS )
    cerr << "TOO MANY TILINGS! " << numTilings << endl;
}


// Clear any trace for feature f      
void ArgumentationAgent::clearTrace( int f)
{
  if ( f > RL_MEMORY_SIZE || f < 0 )
    cerr << "ClearTrace: f out of range " << f << endl;
  if ( traces[ f ] != 0 )
    clearExistentTrace( f, nonzeroTracesInverse[ f ] );
}

// Clear the trace for feature f at location loc in the list of nonzero traces 
void ArgumentationAgent::clearExistentTrace( int f, int loc )
{
  if ( f > RL_MEMORY_SIZE || f < 0 )
    cerr << "ClearExistentTrace: f out of range " << f << endl;
  traces[ f ] = 0.0;
  numNonzeroTraces--;
  nonzeroTraces[ loc ] = nonzeroTraces[ numNonzeroTraces ];
  nonzeroTracesInverse[ nonzeroTraces[ loc ] ] = loc;
}

// Decays all the (nonzero) traces by decay_rate, removing those below minimum_trace 
void ArgumentationAgent::decayTraces( double decayRate )
{
  int f;
  for ( int loc = numNonzeroTraces - 1; loc >= 0; loc-- ) {
    f = nonzeroTraces[ loc ];
    if ( f > RL_MEMORY_SIZE || f < 0 )
      cerr << "DecayTraces: f out of range " << f << endl;
    traces[ f ] *= decayRate;
    if ( traces[ f ] < minimumTrace )
      clearExistentTrace( f, loc );
  }
}

// Set the trace for feature f to the given value, which must be positive   
void ArgumentationAgent::setTrace( int f, float newTraceValue )
{
  if ( f > RL_MEMORY_SIZE || f < 0 )
    cerr << "SetTraces: f out of range " << f << endl;
  if ( traces[ f ] >= minimumTrace )
    traces[ f ] = newTraceValue;         // trace already exists              
  else {
    while ( numNonzeroTraces >= RL_MAX_NONZERO_TRACES )
      increaseMinTrace(); // ensure room for new trace              
    traces[ f ] = newTraceValue;
    nonzeroTraces[ numNonzeroTraces ] = f;
    nonzeroTracesInverse[ f ] = numNonzeroTraces;
    numNonzeroTraces++;
  }
}

// Try to make room for more traces by incrementing minimum_trace by 10%,
// culling any traces that fall below the new minimum                      
void ArgumentationAgent::increaseMinTrace()
{
  minimumTrace *= 1.1;
  cerr << "Changing minimum_trace to " << minimumTrace << endl;
  for ( int loc = numNonzeroTraces - 1; loc >= 0; loc-- ) { // necessary to loop downwards    
    int f = nonzeroTraces[ loc ];
    if ( traces[ f ] < minimumTrace )
      clearExistentTrace( f, loc );
  }
}


void ArgumentationAgent::setParams(int iCutoffEpisodes, int iStopLearningEpisodes) {
  // This function is unused in the published Keepaway code.
  // HandCodedAgent actually exits with error if this function is called.
}


}

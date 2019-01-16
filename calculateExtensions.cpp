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

#define RL_MEMORY_SIZE 1048576
#define RL_MAX_NONZERO_TRACES 100000
#define RL_MAX_NUM_TILINGS 6000



char weightsFile[256];
bool bLearning;
bool bSaveWeights;

/// Hive mind indicator and file descriptor.
bool hiveMind;
int hiveFile;

int epochNum;
int lastAction;

// Added by Niki
#define MAX_STATE_VARS 13
#define MAX_ACTIONS 5

double lastState[ MAX_STATE_VARS ];
enum Situation { Safe, UnderThreat, InDanger };
enum Value { MK, IT, TK };
enum Argument { H, O1, O2, F1, F2 };
enum Label { IN, OUT, UNDEC };

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

// Load / Save weights from/to disk
bool loadWeights( char *filename );
bool saveWeights( char *filename );

// Value function methods for CMACs
int  selectAction();
void initializeTileWidths( int numK, int numT );
double computeQ( int a );
int  argmaxQ();
void updateWeights( double delta );

// Eligibility trace methods
void clearTrace( int f );
void clearExistentTrace( int f, int loc );
void decayTraces( double decayRate );
void setTrace( int f, float newTraceValue );
void increaseMinTrace();

// Support for extra modes and/or analysis.
double getQ(int action);
void setEpsilon(double epsilon);

// Niki-written reward shaping
double getPotential(double state[], int action);
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
double getCorrespondingG(Argument arg, Situation sit);
std::set< std::set<Argument> > getPreferredExtensions(
      std::set<Argument> &args,
      std::set< std::pair<Argument, Argument> > &attacks);
std::set< std::set<Argument> > getPreferredExtensionsFast(
      double state[], std::set<Argument> &args);
std::set<Argument> choosePrefExt(std::set< std::set<Argument> > &prefExts);
int getActionFromExt(std::set<Argument> &prefExt);
double getGFromExt(std::set<Argument> &prefExt, Situation sit);

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


int main(void) {

    srand(time(NULL));

    // Safe, with H O1 O2 F1 F2
    //                 0  1  2  3   4   5  6  7  8  9   10  11  12
    double state[] = { 0, 0, 0, 0, 20,  0, 0, 0, 0, 0,  0,  0,  0 };
    //getPotential(state, 0);
    //std::cout << getPotential(state, 0) << std::endl;
    //std::cout << getPotential(state, 1) << std::endl;
    //std::cout << getPotential(state, 2) << std::endl;
    //std::cout << std::endl;

    // 5 < 10 < 15
    for (double distToT = 4; distToT < 15; distToT += 5) {
        state[3] = distToT;
        for (double minDist1 = 9; minDist1 < 12; minDist1 += 2) {
            state[9] = minDist1;
            for (double minDist2 = 9; minDist2 < 12; minDist2 += 2) {
                state[10] = minDist2;
                for (double minAng1 = 14; minAng1 < 17; minAng1 += 2) {
                    state[11] = minAng1;
                    for (double minAng2 = 14; minAng2 < 17; minAng2 += 2) {
                        state[12] = minAng2;
                        std::cout << "situation: " << getSituation(state) << std::endl;
                        std::set<Argument> args = getApplicableArguments(state);
                        std::cout << "args: ";
                        for (Argument arg : args) {
                            std::cout << arg << " ";
                        }
                        std::cout << std::endl;
                        std::cout << getPotential(state, 0) << " ";
                        std::cout << getPotential(state, 0) << " ";
                        std::cout << getPotential(state, 0) << " ";
                        std::cout << getPotential(state, 0) << " ";
                        std::cout << getPotential(state, 0) << " ";
                        std::cout << getPotential(state, 0) << std::endl << std::endl;
                    }
                }
            }
        }
    }
    
    /*
    // UnderThreat, with H O1 O2 F1 F2
    //                 0  1  2  3   4   5  6  7  8  9   10  11  12
    double state1[] = { 0, 0, 0, 6, 6, 0, 0, 0, 0, 11, 11, 16, 16 };
    getPotential(state1, 0);
    //std::cout << getPotential(state1, 0) << std::endl;
    //std::cout << getPotential(state1, 1) << std::endl;
    //std::cout << getPotential(state1, 2) << std::endl;
    std::cout << std::endl;

    // InDanger, with H O1 O2 F1 F2
    //                 0  1  2  3   4   5  6  7  8  9   10  11  12
    double state2[] = { 0, 0, 0, 4, 4, 0, 0, 0, 0, 11, 11, 16, 16 };
    getPotential(state2, 0);
    //std::cout << getPotential(state2, 0) << std::endl;
    //std::cout << getPotential(state2, 1) << std::endl;
    //std::cout << getPotential(state2, 2) << std::endl;
    
    std::cout << std::endl << std::endl;
    
    // Safe, with H O1 O2 F2
    //                 0  1  2  3   4   5  6  7  8  9   10  11  12
    double state3[] = { 0, 0, 0, 11, 11, 0, 0, 0, 0, 9, 11, 16, 16 };
    getPotential(state3, 0);
    //std::cout << getPotential(state, 0) << std::endl;
    //std::cout << getPotential(state, 1) << std::endl;
    //std::cout << getPotential(state, 2) << std::endl;
    std::cout << std::endl;
    
    // UnderThreat, with H O1 O2 F2
    //                 0  1  2  3   4   5  6  7  8  9   10  11  12
    double state4[] = { 0, 0, 0, 6, 6, 0, 0, 0, 0, 9, 11, 16, 16 };
    getPotential(state4, 0);
    //std::cout << getPotential(state1, 0) << std::endl;
    //std::cout << getPotential(state1, 1) << std::endl;
    //std::cout << getPotential(state1, 2) << std::endl;
    std::cout << std::endl;

    // InDanger, with H O1 O2 F2
    //                 0  1  2  3   4   5  6  7  8  9   10  11  12
    double state5[] = { 0, 0, 0, 4, 4, 0, 0, 0, 0, 9, 11, 16, 16 };
    getPotential(state5, 0);
    //std::cout << getPotential(state2, 0) << std::endl;
    //std::cout << getPotential(state2, 1) << std::endl;
    //std::cout << getPotential(state2, 2) << std::endl;
    */
    
    return 0;
}

double getPotential(double state[], int action) {

    //double start = clock();
    // args will contain all applicable arguments
    std::set<Argument> args = getApplicableArguments(state);
    
    // Get the current situation
    Situation sit = getSituation(state);

    // THIS IS THE STANDARD WAY OF DOING IT, BUT TOO SLOW
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

    // New way of doing extensions

    std::set< std::set<Argument> > prefExts2 = getPreferredExtensionsFast(state, args);
    //double afterPreferredExts = clock();

    
    for (auto el : prefExts2) {
        std::cout << "exts: ";
        for (Argument arg : el) {
            std::cout << arg << " ";
        }
        std::cout << std::endl;
    }
    if (prefExts != prefExts2) {
        std::cout << "----------------------------------------------" << std::endl;
        std::cout << "---" << std::endl;
        for (auto el : prefExts) {
            for (Argument arg : el) {
                std::cout << arg << " ";
            }
            std::cout << std::endl;
        }
    }

    // Randomly choose one of the preferred extensions
    // and get the corresponding action
    std::set<Argument> ext = choosePrefExt(prefExts2);
    int supportedAction = getActionFromExt(ext);
    //double afterActionChosen = clock();

    // If the supported action matches the actual action,
    // return the corresponding potential. Else, return 0.
    //std::cout << "chosen action: " << action << std::endl;
    //std::cout << "recommended action: " << supportedAction << std::endl;
    double toRet = 0;
    /*
    if (action == supportedAction) {
        //std::cout << "match" << std::endl;
        toRet +=  getGFromExt(ext, sit);
    } else {
        //std::cout << "mismatch, pot: 0" << std::endl;
    }
    */
    
    toRet += getGFromExt(ext, sit); // for testing only
    //std::cout << "set attacks " << (setAllAttacks - start)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "simplify " << (afterSimplified - setAllAttacks)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "get preferred " << (afterPreferredExts - afterSimplified)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "get preferred " << (afterPreferredExts - start)/CLOCKS_PER_SEC << std::endl;
    //std::cout << "chose action " << (afterActionChosen - afterPreferredExts)/CLOCKS_PER_SEC << std::endl;
    return toRet;
}

std::set<Argument> getApplicableArguments(double state[]) {
    std::set<Argument> args;
    //enum Argument { H, O1, O2, F1, F2 };
    args.insert(H); // H is always supported

    if (checkOpen(state, 1)) {
        args.insert(O1);
    }
    if (checkOpen(state, 2)) {
        args.insert(O2);
    }
    if (checkFar(state, 1)) {
        args.insert(F1);
    }
    if (checkFar(state, 2)) {
        args.insert(F2);
    }
    return args;
}

std::set< std::pair<Argument, Argument> > setAllAttacks(std::set<Argument> &args, Situation sit) {

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
    // Print out the simplified attacks
    //for (auto attack : attacks) {
        //std::cout << attack.first << "->" << attack.second << std::endl;
    //}
    return attacks;
}

Situation getSituation(double state[]) {
    double minDist = state[3] < state[4] ? state[3] : state[4];

    if (minDist > 10) {
        return Safe;
    } else if (minDist > 5) {
        return UnderThreat;
    } else if (minDist >= 0) {
        return InDanger;
    } else {
        std::cout << "negative distance" << std::endl;
        assert(false);
        return InDanger;
    }
}

void simplifyFramework(std::set< std::pair<Argument, Argument> > &attacks, Situation sit) {

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

Value getValue(Argument arg) {
    if (arg == H) {
        return MK;
    } else if (arg == O1 || arg == O2) {
        return IT;
    } else if (arg == F1 || arg == F2) {
        return TK;
    } else {
        assert(false);
        std::cerr << "wrong argument" << std::endl;
        return MK;
    }
}

double getRelevantPot(Argument arg, Situation sit) {
    Value val = getValue(arg);
    switch (sit) {
        case Safe:
            //std::cout << "Safe" << std::endl;
            switch (val) {
                case MK: return 40;
                case IT: return 20;
                case TK: return 10;
            }
        case UnderThreat:
            //std::cout << "Under Threat" << std::endl;
            switch (val) {
                case MK: return 10;
                case IT: return 20;
                case TK: return 5;
            }
        case InDanger:
            //std::cout << "In Danger" << std::endl;
            switch (val) {
                case MK: return 0;
                case IT: return 25;
                case TK: return 5;
            }
        default:
            assert(false);
            std::cerr << "wrong value" << std::endl;
            return -1;
    }
}

// Get all preferred extensions, with scenario-specific optimization
std::set< std::set<Argument> > getPreferredExtensionsFast(double state[], std::set<Argument> &args) {

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
std::set< std::set<Argument> > getPreferredExtensions(std::set<Argument> &args,
        std::set< std::pair<Argument, Argument> > &attacks) {

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

std::set<Argument> choosePrefExt(std::set< std::set<Argument> > &prefExts) {
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

int getActionFromExt(std::set<Argument> &args) {
    // Get any Argument from the preferred extension (all will promote same action)
    std::set<Argument>::iterator iter = args.begin();
    Argument arg = *iter;
    if (arg == H) {
        return 0;
    } else if (arg == O1 || arg == F1) {
        return 1;
    } else if (arg == O2 || arg == F2) {
        return 2;
    } else {
        assert(false);
        std::cerr << "wrong argument" << std::endl;
        return -1;
    }
}

double getGFromExt(std::set<Argument> &args, Situation sit) {
    double total = 0;
    for (auto arg : args) {
        //std::cout << "argument " << arg << ": " << getRelevantPot(arg, sit) << std::endl;
        total += getRelevantPot(arg, sit);
    }
    return total;
}

bool checkOpen(double state[], int i) {
    if (i == 1) {
        return state[11] >= 15;
    } else if (i == 2) {
        return state[12] >= 15;
    } else {
        assert(false);
        return false;
    }
}

bool checkFar(double state[], int i) {
    if (i == 1) {
        return state[9] >= 10;
    } else if (i == 2) {
        return state[10] >= 10;
    } else {
        assert(false);
        return false;
    }
}

void findLabellings(std::map<Argument, Label> ass,
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

bool isINSubsetOf(std::map<Argument, Label> ass1, std::map<Argument, Label> ass2) {
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

bool containsIllegallyIN(std::map<Argument, Label> &ass,
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

bool isIllegallyIN(Argument arg, std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {
    return ass[arg] == IN && !isLegallyIN(arg, ass, attacks);
}

bool isLegallyIN(Argument arg, std::map<Argument, Label> &ass,
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

bool isSuperIllegallyIN(Argument arg, std::map<Argument, Label> &ass,
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

bool containsSuperIllegallyIN(std::map<Argument, Label> &ass,
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

Argument getSuperIllegallyIN(
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

bool isIllegallyOUT(Argument arg, std::map<Argument, Label> &ass,
        std::set< std::pair<Argument, Argument> > &attacks) {
    return ass[arg] == OUT && !isLegallyOUT(arg, ass, attacks);
}

bool isLegallyOUT(Argument arg, std::map<Argument, Label> &ass,
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

std::map<Argument, Label> transitionStep(std::map<Argument, Label> ass,
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

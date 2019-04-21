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
#include <vector>
#include <set>
#include <cstring>
#include <fstream>
#include <iterator>

// Niki defines
#define DEBUGPRINT      0 // for debugging
#define NUM_ACTIONS     5 // for generalizing
#define NUM_STATE_VARS 25 //  "       "    
//
enum Situation { Safe, UnderThreat, InDanger };
enum Value { MK, IT, TK };
enum Argument { H, O1, O2, O3, O4, F1, F2, F3, F4 };
enum Label { IN, OUT, UNDEC };

extern std::map<std::pair<std::set<Argument>, Situation>, std::set< std::set<Argument> > > myExts;

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

template <class T>
std::set< std::set<T> > getAllSubsets(std::vector<T> s);
void precomputeAllExtensions();

double getRelevantPot(Argument arg, Situation sit);
Value getValue(Argument arg);

std::set<Argument> getApplicableArguments(double state[]);
bool checkOpen(double state[], int i);
bool checkFar(double state[], int i);
std::set< std::pair<Argument, Argument> > setAllAttacks(
      std::set<Argument> &args, Situation sit);
Situation getSituation(double state[]);
void simplifyFramework(
      std::set< std::pair<Argument, Argument> > &attacks,
      Situation sit);

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

std::set< std::set<Argument> > getPreferredExtensions(
      std::set<Argument> &args,
      std::set< std::pair<Argument, Argument> > &attacks);
//
//-----------------------------------------------------------
//

Value getValue(Argument arg) {
    if (arg == H) {
        return MK;
    } else if (arg == O1 || arg == O2 || arg == O3 || arg == O4) {
        return IT;
    } else if (arg == F1 || arg == F2 || arg == F3 || arg == F4) {
        return TK;
    } else {
        assert(false);
        std::cerr << "wrong argument" << std::endl;
        return MK;
    }
}

double getRelevantPot(
        Argument arg, Situation sit) {
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
void simplifyFramework(
        std::set< std::pair<Argument, Argument> > &attacks,
        Situation sit) {

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

// Get all preferred extensions from argumentation framework
std::set< std::set<Argument> > getPreferredExtensions(
        std::set<Argument> &args,
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
std::set<Argument> getApplicableArguments(double state[]) {
    std::set<Argument> args;
    //enum Argument { H, O1, O2, O3, O4, F1, F2, F3, F4 };
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
    if (checkOpen(state, 4)) {
        args.insert(O4);
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
    if (checkFar(state, 4)) {
        args.insert(F4);
    }
    return args;
}

std::set< std::pair<Argument, Argument> > setAllAttacks(
        std::set<Argument> &args, Situation sit) {

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
    if (args.count(O4)) {
        for (auto arg : args) {
            if (arg != O4 && arg != F4) {
                attacks.insert(std::make_pair(O4, arg));
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
    if (args.count(F4)) {
        for (auto arg : args) {
            if (arg != F4 && arg != O4) {
                attacks.insert(std::make_pair(F4, arg));
            }
        }
    }
    // Print out the simplified attacks
    //for (auto attack : attacks) {
        //std::cout << attack.first << "->" << attack.second << std::endl;
    //}
    return attacks;
}

bool checkOpen(double state[], int i) {
    if (i == 1) {
        return state[21] >= 15;
    } else if (i == 2) {
        return state[22] >= 15;
    } else if (i == 3) {
        return state[23] >= 15;
    } else if (i == 4) {
        return state[24] >= 15;
    } else {
        assert(false);
        std::cout << "------checking wrong keeper------" << std::endl;
        return false;
    }
}

bool checkFar(double state[], int i) {
    if (i == 1) {
        return state[17] >= 10;
    } else if (i == 2) {
        return state[18] >= 10;
    } else if (i == 3) {
        return state[19] >= 10;
    } else if (i == 4) {
        return state[20] >= 10;
    } else {
        assert(false);
        std::cout << "------checking wrong keeper------" << std::endl;
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

bool isINSubsetOf(
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

template <class T>
std::set< std::set<T> > getAllSubsets(std::vector<T> set)
{
    std::vector< std::vector<T> > subset;
    std::vector<T> empty;
    subset.push_back( empty );

    for (int i = 0; i < set.size(); i++)
    {
        std::vector< std::vector<T> > subsetTemp = subset;

        for (int j = 0; j < subsetTemp.size(); j++)
            subsetTemp[j].push_back( set[i] );

        for (int j = 0; j < subsetTemp.size(); j++)
            subset.push_back( subsetTemp[j] );
    }
    std::cout << "made it here" << std::endl;

    // Turn it into sets
    std::set< std::set<T> > toReturn;
    for (int i = 0; i < subset.size(); i++) {
        std::set<T> miniToReturn;
        for (int j = 0; j < subset[i].size(); j++) {
            miniToReturn.insert(subset[i][j]);
        }
        toReturn.insert(miniToReturn);
    }
    return toReturn;
    //return subset;
}

void precomputeAllExtensions() {

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
    args.push_back(O4);
    args.push_back(F1);
    args.push_back(F2);
    args.push_back(F3);
    args.push_back(F4);

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
void loadExtensions() {
    const char *name = "exts5v4.txt";

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

void loadSingleExt( std::ifstream &in ) {

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

std::set<Argument> getArgs(std::string str) {
    std::istringstream iss(str);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

//enum Argument { H, O1, O2, O3, O4, F1, F2, F3, F4 };
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
            args.insert(O4);
        } else if (s == "5") {
            args.insert(F1);
        } else if (s == "6") {
            args.insert(F2);
        } else if (s == "7") {
            args.insert(F3);
        } else if (s == "8") {
            args.insert(F4);
        } else {
            std::cerr << "whyyy" << std::endl;
            assert(false);
        }
    }
    return args;
}

Situation getSit(std::string str) {

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

int main(void) {

    //precomputeAllExtensions();
    loadExtensions();
    /*
     *
    std::vector<int> yo;
    yo.push_back(4);
    yo.push_back(3);
    yo.push_back(1);
    yo.push_back(2);

    std::cout << "made it here " << std::endl;
    std::set< std::set<int> > bleh = getAllSubsets(yo);

    for (auto s : bleh) {
        for (auto v : s) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
    */

    return 0;
}

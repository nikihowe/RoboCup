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

#define RL_MEMORY_SIZE 1048576
#define RL_MAX_NONZERO_TRACES 100000
#define RL_MAX_NUM_TILINGS 6000

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
  
  // Added by Niki
  double lastState[ MAX_STATE_VARS ];
  enum Situation { Safe, UnderThreat, InDanger };
  enum Value { Marking, Interception, Tackle };

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
  int  selectAction();
  void initializeTileWidths( int numK, int numT );
  double computeQ( int a );
  int  argmaxQ();
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

  // Niki-written reward shaping
  double getPotential(double state[], int action);
  Situation getSituation(double state[]);
  double getG(Situation sit, Value val);
  bool checkOpen(double state[], int i);
  bool checkFar(double state[], int i);

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
  WorldModel& world,
  int numFeatures, int numActions, bool bLearn,
  double widths[], char *loadWeightsFile,
  char *saveWeightsFile, bool hiveMind
) {
  ArgumentationAgent* agent = new ArgumentationAgent(
    world, numFeatures, numActions, bLearn,
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

double ArgumentationAgent::getPotential(
        double state[], int action) {

    double potential = 0;
    Situation sit = getSituation(state);
    if (action == 0) {
        potential += getG(sit, Marking);
        //std::cout << "hold, reward: " << potential << std::endl;
    } else if (action == 1) {
        if (checkOpen(state, 1)) {
            potential += getG(sit, Interception);
        }
        if (checkFar(state, 1)) {
            potential += getG(sit, Tackle);
        }
        //std::cout << "pass 1, reward: " << potential << std::endl;
    } else if (action == 2) {
        if (checkOpen(state, 2)) {
            potential += getG(sit, Interception);
        }
        if (checkFar(state, 2)) {
            potential += getG(sit, Tackle);
        }
        //std::cout << "pass 2, reward: " << potential << std::endl;
    } else {
        std::cout << "impossible action " << action << std::endl;
        assert(false);
    }
    return 0;
    //return potential / 8.;
}

ArgumentationAgent::Situation ArgumentationAgent::getSituation(
        double state[]) {
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

double ArgumentationAgent::getG(Situation sit, Value val) {
    switch (sit) {
        case Safe:
            //std::cout << "Safe" << std::endl;
            switch (val) {
                case Marking: return 40;
                case Interception: return 20;
                case Tackle: return 10;
            }
        case UnderThreat:
            //std::cout << "Under Threat" << std::endl;
            switch (val) {
                case Marking: return 10;
                case Interception: return 20;
                case Tackle: return 5;
            }
        case InDanger:
            //std::cout << "In Danger" << std::endl;
            switch (val) {
                case Marking: return 0;
                case Interception: return 25;
                case Tackle: return 5;
            }
        default: return 0;
    }
}

bool ArgumentationAgent::checkOpen(double state[], int i) {
    if (i == 1) {
        return state[11] >= 15;
    } else if (i == 2) {
        return state[12] >= 15;
    } else {
        assert(false);
        return false;
    }
}

bool ArgumentationAgent::checkFar(double state[], int i) {
    if (i == 1) {
        return state[9] >= 10;
    } else if (i == 2) {
        return state[10] >= 10;
    } else {
        assert(false);
        return false;
    }
}

int ArgumentationAgent::startEpisode( double state[] )
{
  if (hiveMind) loadColTabHeader(colTab, weights);
  epochNum++;
  decayTraces( 0 );
  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    Q[ a ] = computeQ( a );
  }

  lastAction = selectAction();

  char buffer[128];
  sprintf( buffer, "Q[%d] = %.2f", lastAction, Q[lastAction] );
  LogDraw.logText( "Qmax", VecPosition( 25, -30 ),
                   buffer,
                   1, COLOR_BROWN );

  for ( int j = 0; j < numTilings; j++ )
    setTrace( tiles[ lastAction ][ j ], 1.0 );
  if (hiveMind) saveWeights(weightsFile);

  // Niki added
  for (int i = 0; i < MAX_STATE_VARS; i++) {
      lastState[i] = state[i];
  }
  return lastAction;
}


int ArgumentationAgent::step( double reward, double state[] )
{
  // Niki-written
  double oldPotential = 0;
  if (bLearning) {
    oldPotential = getPotential(lastState, lastAction);
  }
  //std::cout << getSituation(lastState) << " " << lastAction << " -> ";

  if (hiveMind) loadColTabHeader(colTab, weights);
  double delta = reward - Q[ lastAction ];
  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    Q[ a ] = computeQ( a );
  }

  lastAction = selectAction();

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

  // Niki-written
  double newPotential = getPotential(state, lastAction);
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
  for (int i = 0; i < MAX_STATE_VARS; i++) {
      lastState[i] = state[i];
  }
  return lastAction;
}


void ArgumentationAgent::endEpisode( double reward )
{
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

    // Niki-written
    double oldPotential = getPotential(lastState, lastAction);
    delta -= oldPotential; // newPotential==0
    //std::cout << std::endl << std::flush;
    updateWeights( delta );
    // TODO Actually, there's still possibly risk for trouble here with multiple
    // TODO players stomping each other. Is this okay?
    // TODO The weight updates themselves are in order.
  }
  if ( bLearning && bSaveWeights && rand() % 200 == 0 && !hiveMind ) {
    saveWeights( weightsFile );
  }
  if (hiveMind) saveWeights(weightsFile);
  lastAction = -1;
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

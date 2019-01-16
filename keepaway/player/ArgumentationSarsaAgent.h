#ifndef ARGUMENTATION_SARSA_AGENT
#define ARGUMENTATION_SARSA_AGENT

#include <cassert>
#include "SMDPAgent.h"
#include "tiles2.h"

#define RL_MEMORY_SIZE 1048576
#define RL_MAX_NONZERO_TRACES 100000
#define RL_MAX_NUM_TILINGS 6000

class ArgumentationSarsaAgent:public SMDPAgent
{
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
  ArgumentationSarsaAgent           ( int    numFeatures,
                                      int    numActions,
                                      bool   bLearn,
                                      double widths[],
                                      char   *loadWeightsFile,
                                      char   *saveWeightsFile,
                                      bool   hiveMind);

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
} ;

#endif

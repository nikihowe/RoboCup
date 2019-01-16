/*
Copyright (c) 2004 Gregory Kuhlmann, Peter Stone
University of Texas at Austin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the University of Amsterdam nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <assert.h>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iostream>

#include "WorldModel.h"

int WorldModel::getNumKeepers()
{
  return m_numKeepers;
}

void WorldModel::setNumKeepers( int iNum )
{
  m_numKeepers = iNum;
}

int WorldModel::getNumTakers()
{
  return m_numTakers;
}

void WorldModel::setNumTakers( int iNum )
{
  m_numTakers = iNum;
}

double WorldModel::congestion( VecPosition pos, bool considerMe ) 
{
  double congest = 0;
  if ( considerMe && pos != getAgentGlobalPosition() ) 
    congest = 1 / getAgentGlobalPosition().getDistanceTo( pos );

  VecPosition playerPos;

  int iIndex;
  for( ObjectT obj = iterateObjectStart( iIndex, OBJECT_SET_PLAYERS );
       obj != OBJECT_ILLEGAL;
       obj = iterateObjectNext ( iIndex, OBJECT_SET_PLAYERS ) ) {
    if ( ( playerPos = getGlobalPosition( obj ) ) != pos )
      if ( obj != getAgentObjectType() )
      /* Don't want to count a player in its own congestion measure */
      congest += 1/playerPos.getDistanceTo( pos );
  }

  return congest;
}

void WorldModel::resetEpisode()
{
  LogDraw.logText( "episode", VecPosition( 0, 0 ), "Reset", 
		      40, COLOR_WHITE );
  Ball.setTimeLastSeen( -1 );
  for ( int i = 0; i < MAX_TEAMMATES; i++ )
    Teammates[i].setTimeLastSeen( -1 );
  for ( int i = 0; i < MAX_OPPONENTS; i++ )
    Opponents[i].setTimeLastSeen( -1 );
  for ( int i = 0; i < MAX_TEAMMATES+MAX_OPPONENTS; i++ )
    UnknownPlayers[i].setTimeLastSeen( -1 );
  iNrUnknownPlayers = 0;
  for ( int i = 0; i < MAX_FLAGS; i++ )
    Flags[i].setTimeLastSeen( -1 );
  for ( int i = 0; i < MAX_LINES; i++ )
    Lines[i].setTimeLastSeen( -1 );

  setNewEpisode( true );
}

void WorldModel::setNewEpisode( bool bNewEp )
{
  m_newEpisode = bNewEp;
}

bool WorldModel::isNewEpisode()
{
  return m_newEpisode;
}

double WorldModel::keeperReward()
{
  double reward = getCurrentCycle() - getTimeLastAction();
  return reward;
}

void WorldModel::setLastAction( int iAction )
{
  m_lastAction = iAction;
  m_timeLastAction = 
    ( iAction == UnknownIntValue ) ? UnknownTime : getCurrentCycle();
}

int WorldModel::getLastAction()
{
  return m_lastAction;
}

// Niki-added
void WorldModel::setLastGlobalAction( int action ) {
    std::ofstream outfile;
    outfile.open("action.dat");
    outfile << action;
    outfile.close();
}

int WorldModel::getLastGlobalAction() {

    int toRet;
    std::ifstream infile("action.dat", std::ios::in);

    if (!infile.is_open()) {
        std::cerr << "Probelm opening file" << std::endl;
        assert(false);
    }

    infile >> toRet;
    return toRet;
}

void WorldModel::setLastGlobalState( double state[] )
{
    
    // We ran this for ~16h and it had no problems,
    // so I think it's pretty safe to use
    std::ofstream outfile;
    outfile.open("state.dat");
    for (int i = 0; i < 13; ++i) {
        outfile << state[i] << " ";
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

std::vector<double> WorldModel::getLastGlobalState()
{
    std::vector<double> toRet;
    std::ifstream infile("state.dat", std::ios::in);

    if (!infile.is_open()) {
        std::cerr << "Probelm opening file" << std::endl;
        assert(false);
    }
    double num = 0;
    while (infile >> num && toRet.size() < 13) {
        toRet.push_back(num);
    }
    assert(toRet.size() == 13);

    return toRet;
    
    //std::vector<double> bleh(13, 0);
    //return bleh;
    //std::cout << "getting last state: ";
    //for (int i = 0; i < 13; i++) {
        //std::cout << m_lastState[i] << " ";
    //}
    //std::cout << std::endl;
    //return m_lastState;
}

int WorldModel::getTimeLastDecision() { // Niki
    return timeLastDecision;
}

void WorldModel::setTimeLastDecision( int time ) { // Niki
    timeLastDecision = time;
}

int WorldModel::getTimeLastAction()
{
  return m_timeLastAction;
}

// Fills the provided array with state information based on world
int WorldModel::keeperStateVars( double state[] )
{
    // with Niki-added comments
    // Identify the teammate with the ball
  ObjectT PB = getClosestInSetTo( OBJECT_SET_TEAMMATES, OBJECT_BALL );
  if ( !SoccerTypes::isTeammate( PB ) )
    return 0;

  VecPosition C = getKeepawayRect().getPosCenter();

  double WB_dist_to_C = getGlobalPosition( PB ).getDistanceTo( C );
  
  int numK = getNumKeepers();
  int numT = getNumTakers();

  // Get a list of all other keepers, unordered
  ObjectT K[ numK ];
  for ( int i = 0; i < numK; i++ )
    K[ i ] = SoccerTypes::getTeammateObjectFromIndex( i );

  // Get a list of all takers, unordered
  ObjectT T[ numT ];
  for ( int i = 0; i < numT; i++ )
    T[ i ] = SoccerTypes::getOpponentObjectFromIndex( i );

  // Sort keepers from closest to furthest from PB
  double WB_dist_to_K[ numK ];
  if ( !sortClosestTo( K, numK, PB, WB_dist_to_K ) )
    return 0;
  
  // Sort takers from closest to furthest from PB
  double WB_dist_to_T[ numT ];
  if ( !sortClosestTo( T, numT, PB, WB_dist_to_T ) )
    return 0;

  // Get keeper 1 and 2 distances to center
  double dist_to_C_K[ numK ];
  for ( int i = 1; i < numK; i++ ) {
    dist_to_C_K[ i ] = getGlobalPosition( K[ i ] ).getDistanceTo( C );
  }

  // Get taker 0 and 1 distances to center
  double dist_to_C_T[ numT ];
  for ( int i = 0; i < numT; i++ ) {
    dist_to_C_T[ i ] = getGlobalPosition( T[ i ] ).getDistanceTo( C );
  }

  // Get distance of K to nearest opponent
  double nearest_Opp_dist_K[ numK ];
  VecPosition posPB = getGlobalPosition( PB );
  for ( int i = 1; i < numK; i++ ) {
    VecPosition pos = getGlobalPosition( K[ i ] );
    for ( int j = 0; j < numT; j++ ) {
      double tmp = getGlobalPosition( T[ j ] ).getDistanceTo( pos );
      if ( j == 0 || tmp < nearest_Opp_dist_K[ i ] ) {
        nearest_Opp_dist_K[ i ] = tmp;
      }
    }
  }

  double nearest_Opp_ang_K[ numK ];
  for ( int i = 1; i < numK; i++ ) {
    VecPosition pos = getGlobalPosition( K[ i ] );
    for ( int j = 0; j < numT; j++ ) {
      double tmp = posPB.getAngleBetweenPoints( pos, getGlobalPosition( T[ j ] ) );
      if ( j == 0 || tmp < nearest_Opp_ang_K[ i ] ) {
        nearest_Opp_ang_K[ i ] = tmp;
      }
    }
  }

  int j = 0;
  state[ j++ ] = WB_dist_to_C; // 0: distance to center
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = WB_dist_to_K[ i ]; // 1,2: distance of K1, K2 to K0
  for ( int i = 0; i < numT; i++ )
    state[ j++ ] = WB_dist_to_T[ i ]; // 3,4: distance of T0, T1 to K0
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = dist_to_C_K[ i ]; // 5,6: distance of K1, K2 to C
  for ( int i = 0; i < numT; i++ )
    state[ j++ ] = dist_to_C_T[ i ]; // 7,8: distance of T0, T1 to C
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = nearest_Opp_dist_K[ i ]; // 9,10: dist of K1, K2 to T_near
  for ( int i = 1; i < numK; i++ )
    state[ j++ ] = nearest_Opp_ang_K[ i ]; // 11,12: ang of K1, K2 to T_small

  return j; // should be 13
}

// Greg:
// This really doesn't belong here,
// because it is related to a specific 
// learner.  I don't know exactly
// where to put it, though because
// I want to keep the LinearSarsa
// class generic.

// Yaxin: changed from keeperTileWidths to keeperResolutions and keeperRanges,

int WorldModel::keeperStateRangesAndResolutions( double ranges[], 
						 double minValues[], 
						 double resolutions[], 
						 int numK, int numT )
{
  if ( numK < 3 ) {
    cerr << "keeperTileWidths: num keepers must be at least 3, found: " 
	 << numK << endl;
    return 0;
  }
  
  if ( numT < 2 ) {
    cerr << "keeperTileWidths: num takers must be at least 2, found: " 
	 << numT << endl;
    return 0;
  }

  int j = 0;

  double maxRange = hypot( 25, 25 );
			  
  ranges[ j ] = maxRange / 2.0;        // WB_dist_to_center           
  minValues[ j ] = 0;
  resolutions[ j++ ] = 2.0; 
  for ( int i = 1; i < numK; i++ ) {     // WB_dist_to_T          
    ranges[ j ] = maxRange;
    minValues[ j ] = 0;
    resolutions[ j++ ] = 2.0 + ( i - 1 ) / ( numK - 2 );
  }
  for ( int i = 0; i < numT; i++ ) {     // WB_dist_to_O
    ranges[ j ] = maxRange;
    minValues[ j ] = 0;
    resolutions[ j++ ] = 3.0 + ( i - 1 ) / ( numT - 1 );
  }
  for ( int i = 1; i < numK; i++ ) {     // dist_to_center_T    
    ranges[ j ] = maxRange / 2.0;
    minValues[ j ] = 0;
    resolutions[ j++ ] = 2.0 + ( i - 1 ) / ( numK - 2 );
  }
  for ( int i = 0; i < numT; i++ ) {     // dist_to_center_O  
    ranges[ j ] = maxRange / 2.0;
    minValues[ j ] = 0;
    resolutions[ j++ ] = 3.0;
  }
  for ( int i = 1; i < numK; i++ ) {     // nearest_Opp_dist_T 
    ranges[ j ] = maxRange / 2.0;
    minValues[ j ] = 0;
    resolutions[ j++ ] = 4.0;
  }
  for ( int i = 1; i < numK; i++ ) {     // nearest_Opp_ang_T  
    ranges[ j ] = 180;
    minValues[ j ] = 0;
    resolutions[ j++ ] = 10.0;
  }

  return j;
}


void WorldModel::setMoveSpeed( double speed )
{
  m_moveSpeed = speed;
}

double WorldModel::getMoveSpeed()
{
  return m_moveSpeed;
}

void WorldModel::setKeepawayRect( VecPosition pos1, VecPosition pos2 )
{
  m_keepawayRect.setRectanglePoints( pos1, pos2 );
}

Rect WorldModel::getKeepawayRect()
{
  return m_keepawayRect;
}


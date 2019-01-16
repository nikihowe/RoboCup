int ArgumentationAgent::startEpisode( double state[] )
{
  if (hiveMind) loadColTabHeader(colTab, weights);
  epochNum++;
  decayTraces( 0 );
  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    Q[ a ] = computeQ( a ); // calculate Q for this state, across all actions
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
  double start = clock();
  // Niki-written
  double oldPotential = 0;
  if (bLearning) {
    oldPotential = getPotential(lastState, lastAction);
  }

  if (hiveMind) loadColTabHeader(colTab, weights);
  double delta = reward - Q[ lastAction ];
  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    Q[ a ] = computeQ( a ); // calculate Q for this state, across all actions
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
  // ^ this is Q(s', a') because we recalculated in 955

  // Niki-written
  double newPotential = getPotential(state, lastAction);
  delta += newPotential - oldPotential; // Assumes gamma==1

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
  double end = clock();
  
  if ((end - start)*1.0/CLOCKS_PER_SEC >= 0.0125) {
    std::cerr << "too slow" << std::endl;
    std::cerr << "---------------------------------------------------------------" << std::endl;
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

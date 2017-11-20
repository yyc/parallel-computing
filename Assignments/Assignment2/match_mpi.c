#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <time.h>

// Message format: int array. The first index denotes the type of the message:
// only from FP0, [rank, xPosition, yPosition]
#define GAME_START 0

// only from FP0, [ballX, ballY, rank] (the rank of the FP to handle challenges)
#define ROUND_START 1

// From each player, [xPosition, yPosition, ballChallenge]
#define PLAYER_MOVE 2

// From the FP holding the ball (FPX): [rank]
#define KICK_PHASE 3

// Either from the ball winner or from the FPX: [ballX, ballY]
#define ROUND_OVER 5

#define DEFAULT_TAG 0

#define MSG_LENGTH 5

#define NUMPLAYERS 2
#define NUMROUNDS 10
#define FIELD_WIDTH 64
#define FIELD_LENGTH 128

struct playerInfo {
  int team;
  int number;
  int rank;
  int posX;
  int posY;
  int newX;
  int newY;
  int challenge;
};

void rootBroadcast(int *buffer) {
  MPI_Bcast(buffer,
            MSG_LENGTH,
            MPI_INT,
            0,
            MPI_COMM_WORLD);
}

void broadcast(int *buffer, int broadcaster) {
  MPI_Bcast(buffer,
            MSG_LENGTH,
            MPI_INT,
            broadcaster,
            MPI_COMM_WORLD);
}

int rankFor(int posX, int posY) {
  return 0;
}

// Master and field program.
void field(int rank) {
  int  i = 0;
  bool isRoot = rank == 0;
  int  currentField, highest_challenge, current_winner;
  int  numplayers = NUMPLAYERS;
  int  rounds = 0;
  int *buffer = (int *)malloc((sizeof(int) * MSG_LENGTH));
  int  ballX, ballY, numWithBall, baller;

  // printf("Initiating Master Field Program..\n");

  struct playerInfo players[numplayers];


  // Initialize every player, send them their info
  for (i = 0; i < numplayers; i++) {
    if (isRoot) {
      // since the field has rank 0, and players are from 1-11
      players[i].rank = 12 + i;
      players[i].posX = rand() % FIELD_WIDTH;
      players[i].posY = rand() % FIELD_LENGTH;

      buffer[0] = GAME_START;
      buffer[1] = players[i].rank;
      buffer[2] = players[i].posX;
      buffer[3] = players[i].posY;
      printf("Sending out for %i\n", players[i].rank);
    }

    // Broadcast
    rootBroadcast(buffer);
  }

  if (rank == 0) {
    ballX = rand() % FIELD_WIDTH;
    ballY = rand() % FIELD_LENGTH;
  }


  // Run NUMROUNDS times
  while (++rounds <= NUMROUNDS) {
    if (isRoot) {
      printf("%i\n%i %i\n",
             rounds,
             ballY,
             ballX);

      // Start round, send out ball location
      buffer[0] = ROUND_START;
      buffer[1] = ballX;
      buffer[2] = ballY;
      buffer[3] = rankFor(ballX, ballY);
    }

    // ROUND_START
    rootBroadcast(buffer);
    ballX             = buffer[1];
    ballY             = buffer[2];
    currentField      = buffer[3];
    highest_challenge = -1;
    current_winner    = -1;

    // Receive player data
    for (int i = 0; i < NUMPLAYERS; i++) {
      broadcast(buffer, 12 + i);

      if (isRoot) {
        players[i].newX = buffer[1];
        players[i].newY = buffer[2];

        if ((buffer[1] == ballX) && (buffer[2] == ballY)) {
          players[i].challenge = buffer[3];
        } else {
          players[i].challenge = -1;
        }
      }

      // We're responsible for handling the challenges
      if (currentField == rank) {
        buffer[0] = KICK_PHASE;

        // player has reached
        if ((buffer[1] == ballX) && (buffer[2] == ballY)) {
          if (buffer[3] > highest_challenge) {
            current_winner = i;
          }
        }

        if (current_winner == -1) {
          buffer[1] = currentField;
        } else {
          buffer[1] = current_winner + 12;
        }
      }
    }
    broadcast(buffer, currentField);

    if (isRoot) {
      // We can print the round information
      for (i = 0; i < NUMPLAYERS; i++) {
        printf("%i %i %i %i %i %i %i %i\n",
               i % 11,                         // Player
                                               // number
                                               // (within the
                                               // team)
               players[i].posY,                // starting
                                               // position
               players[i].posX,
               players[i].newY,                // ending
                                               // position
               players[i].newX,
               (players[i].newX == ballX) &&   // reached
               (players[i].newY == ballY),
               (players[i].rank == buffer[1]), // kicked
               players[i].challenge            // challenge
               );
        players[i].posX = players[i].newX;
        players[i].posY = players[i].newY;
      }
    }

    // if nobody won, the field sends out ROUND_OVER
    if (buffer[1] == rank) {
      buffer[0] = ROUND_OVER;
      buffer[1] = ballX;
      buffer[2] = ballY;
      broadcast(buffer, rank);
    } else {
      broadcast(buffer, buffer[1]);
    }

    if (isRoot) {
      ballX = buffer[1];
      ballY = buffer[2];
    }
  }
}

// Moves posX and posY 10 steps towards the goal
// Returns true if we reach the ball;
int moveTowards(int *xPos, int *yPos, int ballX, int ballY, int limit) {
  int x = *xPos;
  int y = *yPos;

  for (int i = 0; i < limit; i++) {
    if (ballX > x) {
      x++;
    } else if (ballX < x) {
      x--;
    } else if (ballY > y) {
      y++;
    } else if (ballY < y) {
      y--;
    } else {
      *xPos = x;
      *yPos = y;
      return i;
    }
  }
  *xPos = x;
  *yPos = y;

  return limit;
}

int challenge(int dribble) {
  int r = 1 + (rand() % 10);

  return r * dribble;
}

// Player program. Always using blocking sends and receives
void player(int rank) {
  printf("Player with rank %i started\n", rank);
  int type, posX, posY, ballX, ballY, myIndex, currentField;

  int rounds    = 0;
  int speed     = 14;
  int dribbling = 1;
  int kick      = 0;
  MPI_Status stat;

  int *buffer = (int *)malloc((sizeof(int) * MSG_LENGTH));

  // Get all the stuff at the start of the game
  for (int i = 0; i < NUMPLAYERS; i++) {
    rootBroadcast(buffer);

    if (rank == buffer[1]) {
      posX    = buffer[2];
      posY    = buffer[3];
      myIndex = i;

      // printf("Player %i Reporting In! I'm at % i, % i \n", rank, posX,
      // posY);
    }
  }

  while (++rounds <= NUMROUNDS) {
    // ROUND_START
    rootBroadcast(buffer);
    type = buffer[0];

    ballX        = buffer[1];
    ballY        = buffer[2];
    currentField = buffer[3];

    // move towards the ball or die trying
    moveTowards(&posX, &posY, ballX, ballY, speed);

    printf("Player %i moved to % i, % i \n", rank, posX, posY);

    for (int i = 0; i < NUMPLAYERS; i++) {
      if (i == myIndex) {
        buffer[0] = PLAYER_MOVE;
        buffer[1] = posX;
        buffer[2] = posY;

        if ((posX == ballX) && (posY == ballY)) {
          buffer[3] = challenge(dribbling);
        }
      }
      broadcast(buffer, 12 + i);
    }

    // KICK_PHASE
    broadcast(buffer, currentField);

    if (rank == buffer[1]) {
      buffer[0] = ROUND_OVER;
      buffer[1] = rand() % FIELD_WIDTH;
      buffer[2] = rand() % FIELD_LENGTH;

      // ROUND_END
      broadcast(buffer, rank);
    } else {
      broadcast(buffer, buffer[1]);
    }
  }
}

int main(int argc, char *argv[])
{
  int  numtasks, rank;
  int *msg;


  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Make random numbers different for each thread
  srand((unsigned)time(NULL) + rank);

  if (rank < 12)
  { // Process is a field
    field(rank);
  }
  else
  { // Initialize player
    player(rank);
  }
  MPI_Finalize();
}

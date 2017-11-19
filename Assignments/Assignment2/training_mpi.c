#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <time.h>

#define GAME_START 0
#define ROUND_START 1
#define KICK_PHASE 2
#define GAME_OVER 3

#define DEFAULT_TAG 0

#define MSG_LENGTH 6
#define NUMROUNDS 900
#define FIELD_WIDTH 64
#define FIELD_LENGTH 128

/* Message format: int array
   Field 1: Type, either GAME_START, ROUND_START, KICK_PHASE or GAME_OVER
   Depending on the type, the remaining fields:
   GAME_START: [playerX, playerY]
   ROUND_START (from master): [ballX, ballY];
   ROUND_START (from player: [playerX, playerY, distanceRan, numReached,
      numKicked]
 */

struct playerInfo {
  int         posX;
  int         posY;
  int         rank;
  int        *messageBuffer;
  MPI_Request request;
  MPI_Status  stat;
};

// does an Isend to the person with the player's messageBuffer
void IsendPlayer(struct playerInfo *player) {
  MPI_Isend(player->messageBuffer, MSG_LENGTH, MPI_INT, player->rank,
            DEFAULT_TAG, MPI_COMM_WORLD, &(player->request));
}

// Blocking Receives a response from that player
void RecvPlayer(struct playerInfo *player) {
  MPI_Recv(player->messageBuffer, MSG_LENGTH, MPI_INT, player->rank,
           DEFAULT_TAG, MPI_COMM_WORLD, &(player->stat));
}

// Master and field program. Always uses non-blocking sending and blocking
// receiving
void master(int numtasks) {
  int  i = 0;
  int  numplayers = numtasks - 1;
  int  rounds = 0;
  int *buffer = (int *)malloc((sizeof(int) * MSG_LENGTH));
  int  ballX, ballY, numWithBall, baller;

  // printf("Initiating Master Field Program..\n");

  struct playerInfo players[numplayers];

  // Initialize every player, send them their info
  for (i = 0; i < numplayers; i++) {
    // since the field has rank 0, and players are from 1-11
    players[i].rank             = i + 1;
    players[i].posX             = rand() % FIELD_WIDTH;
    players[i].posY             = rand() % FIELD_LENGTH;
    players[i].messageBuffer    = (int *)malloc((sizeof(int) * MSG_LENGTH));
    players[i].messageBuffer[0] = GAME_START;
    players[i].messageBuffer[1] = players[i].posX;
    players[i].messageBuffer[2] = players[i].posY;
    IsendPlayer(&players[i]);
  }

  ballX = rand() % FIELD_WIDTH;
  ballY = rand() % FIELD_LENGTH;

  // Run NUMROUNDS times
  while (++rounds <= NUMROUNDS) {
    printf("%i\n%i %i\n",
           rounds,
           ballY,
           ballX);

    // Start round, send out ball location
    for (i = 0; i < numplayers; i++) {
      players[i].messageBuffer[0] = ROUND_START;
      players[i].messageBuffer[1] = ballX;
      players[i].messageBuffer[2] = ballY;
      IsendPlayer(&players[i]);
    }

    // Receives each player's movement
    numWithBall = 0;

    for (i = 0; i < numplayers; i++) {
      RecvPlayer(&players[i]);

      if ((players[i].messageBuffer[1] == ballX) &&
          (players[i].messageBuffer[2] == ballY)) {
        numWithBall++;

        // Simple one-pass random sample.
        // THe first gets chosen with p = 1, the second replaces that with p =
        // 1/2, and so on
        if (rand() % numWithBall == 0) {
          baller = i;
        }
      }
    }

    for (i = 0; i < numplayers; i++) {
      printf("%i %i %i %i %i %i %i %i %i %i\n",
             i,
             players[i].posX,
             players[i].posY,
             players[i].messageBuffer[1],
             players[i].messageBuffer[2],
             (players[i].messageBuffer[1] == ballX) &&
             (players[i].messageBuffer[2] == ballY),
             (numWithBall != 0) && (i == baller),
             players[i].messageBuffer[3],
             players[i].messageBuffer[4],
             players[i].messageBuffer[5]
             );
      players[i].posX = players[i].messageBuffer[1];
      players[i].posY = players[i].messageBuffer[2];
    }

    // Skip the kick phase if noboody has the ball
    if (numWithBall == 0) {
      continue;
    }

    // If a player has the ball, enter the kick phase.
    players[baller].messageBuffer[0] = KICK_PHASE;
    IsendPlayer(&players[baller]);
    RecvPlayer(&players[baller]);
    ballX = players[baller].messageBuffer[1];
    ballY = players[baller].messageBuffer[2];

    // printf("Player %i kicked the ball to %i %i!\n",
    //        players[baller].rank,
    //        ballX,
    //        ballY);
  }

  // printf("Game Over, Man. Game Over.\n");

  // Tell all players the game is up
  for (i = 0; i < numplayers; i++) {
    players[i].messageBuffer[0] = GAME_OVER;
    IsendPlayer(&players[i]);
  }
}

// Moves posX and posY 10 steps towards the goal
// Returns true if we reach the ball;
int moveTowards(int *xPos, int *yPos, int ballX, int ballY) {
  int x = *xPos;
  int y = *yPos;

  for (int i = 0; i < 10; i++) {
    if (ballX > x) {
      x++;
    } else if (ballX < x) {
      x--;
    } else if (ballY > y) {
      y++;
    } else if (ballY < y) {
      y--;
    } else {
      return i;
    }
  }
  *xPos = x;
  *yPos = y;

  return 10;
}

// Player program. Always using blocking sends and receives
void player(int rank) {
  int type, posX, posY, ballX, ballY;

  int ran          = 0;
  int timesReached = 0;
  int timesKicked  = 0;
  MPI_Status stat;

  int *buffer = (int *)malloc((sizeof(int) * MSG_LENGTH));

  do {
    MPI_Recv(buffer, MSG_LENGTH, MPI_INT, 0, DEFAULT_TAG, MPI_COMM_WORLD, &stat);
    type = buffer[0];

    if (type == GAME_OVER) {
      continue;
    } else if (type == GAME_START) {
      posX = buffer[1];
      posY = buffer[2];

      // printf("Player %i Reporting In! I'm at % i, % i \n", rank, posX,
      // posY);
    } else if (type == ROUND_START) {
      ballX = buffer[1];
      ballY = buffer[2];

      // move towards the ball or die trying
      ran += moveTowards(&posX, &posY, ballX, ballY);

      // printf("Player %i moved to % i, % i \n", rank, posX, posY);

      if ((posX == ballX) && (posY == ballY)) {
        timesReached++;
      }
      buffer[1] = posX;
      buffer[2] = posY;
      buffer[3] = ran;
      buffer[4] = timesReached;
      buffer[5] = timesKicked;
      MPI_Send(buffer, MSG_LENGTH, MPI_INT, 0, DEFAULT_TAG, MPI_COMM_WORLD);
    } else if (type == KICK_PHASE) {
      timesKicked++;
      buffer[1] = rand() % FIELD_WIDTH;
      buffer[2] = rand() % FIELD_LENGTH;
      MPI_Send(buffer, MSG_LENGTH, MPI_INT, 0, DEFAULT_TAG, MPI_COMM_WORLD);
    }
    ballX = 0;
    ballY = 0;
  } while (type != GAME_OVER);

  // printf(
  //   "Player %i Ending.. I ran %i metres, reached the ball %i times and kicked
  // %i times.\n",
  //   rank,
  //   ran,
  //   timesReached,
  //   timesKicked);
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


  if (rank == 0)
  { // Process is the leader
    master(numtasks);
  }
  else
  {
    player(rank);
  }
  MPI_Finalize();
}

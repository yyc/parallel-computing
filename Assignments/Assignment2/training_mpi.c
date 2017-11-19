#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define GAME_START 0
#define ROUND_START 1
#define KICK_PHASE 2
#define GAME_OVER 3

#define DEFAULT_TAG 0

#define FIELD_WIDTH 64
#define FIELD_LENGTH 128

/* Message format: 3-int array
   Field 1: Type, either GAME_START, ROUND_START, KICK_PHASE or GAME_OVER
   Depending on the type, the remaining fields:
   GAME_START: [playerX, playerY]
   ROUND_START: [ballX, ballY];
 */

struct playerInfo {
  int         posX;
  int         posY;
  int         rank;
  int        *messageBuffer;
  MPI_Request request;
};

// does an Isend to the person with the player's messageBuffer
void IsendPlayer(struct playerInfo *player) {
  MPI_Isend(player->messageBuffer, 3, MPI_INT, player->rank,
            DEFAULT_TAG, MPI_COMM_WORLD, &(player->request));
}

// Master and field program. Always uses non-blocking sending and blocking
// receiving
void master(int numtasks) {
  int  i = 0;
  int  numplayers = numtasks - 1;
  int  rounds = 0;
  int *buffer = (int *)malloc((sizeof(int) * 3));
  int  ballX, ballY;

  printf("Initiating Master Field Program..\n");

  struct playerInfo players[numplayers];

  // Initialize every player, send them their info
  for (i = 0; i < numplayers; i++) {
    // since the field has rank 0, and players are from 1-11
    players[i].rank             = i + 1;
    players[i].posX             = rand() % FIELD_WIDTH;
    players[i].posY             = rand() % FIELD_LENGTH;
    players[i].messageBuffer    = (int *)malloc((sizeof(int) * 3));
    players[i].messageBuffer[0] = GAME_START;
    players[i].messageBuffer[1] = players[i].posX;
    players[i].messageBuffer[2] = players[i].posY;
    IsendPlayer(&players[i]);
  }

  ballX = rand() % FIELD_WIDTH;
  ballY = rand() % FIELD_LENGTH;

  while (++rounds <= 2) {
    printf("Master: =====ROUND %i START=====", rounds);

    for (i = 0; i < numplayers; i++) {
      players[i].messageBuffer[0] = ROUND_START;
      players[i].messageBuffer[1] = ballX;
      players[i].messageBuffer[2] = ballY;
      IsendPlayer(&players[i]);
    }
  }

  // Tell all players the game is up
  for (i = 0; i < numplayers; i++) {
    players[i].messageBuffer[0] = GAME_OVER;
    IsendPlayer(&players[i]);
  }
}

// Player program. Always using blocking sends and receives
void player(int rank) {
  int type, posX, posY, ballX, ballY;
  MPI_Status stat;

  int *buffer = (int *)malloc((sizeof(int) * 3));

  do {
    MPI_Recv(buffer, 3, MPI_INT, 0, DEFAULT_TAG, MPI_COMM_WORLD, &stat);
    type = buffer[0];

    if (type == GAME_OVER) {
      continue;
    } else if (type == GAME_START) {
      posX = buffer[1];
      posY = buffer[2];
      printf("Player %i Reporting In! I'm at % i, % i \n", rank, posX, posY);
    } else if (type == ROUND_START) {
      ballX = buffer[1];
      ballY = buffer[2];
      printf("New Round, ball at %i %i\n", ballX, ballY);
    } else if (type == KICK_PHASE) {}
    ballX = 0;
    ballY = 0;
  } while (type != GAME_OVER);
  printf("Player %i Ending..\n", rank);
}

int main(int argc, char *argv[])
{
  int  numtasks, rank;
  int *msg;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

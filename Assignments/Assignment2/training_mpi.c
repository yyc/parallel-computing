#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define GAME_START 0
#define ROUND_START 1
#define KICK_PHASE 2
#define GAME_OVER 3

/* Message format: 3-int array
   Field 1: Type, either GAME_START, ROUND_START, KICK_PHASE or GAME_OVER
   Depending on the type, the remaining fields:
   GAME_START: [playerX, playerY]
   ROUND_START: [ballX, ballY];
 */

struct playerInfo {
  int posX;
  int posY;
  int rank;
}

// Master and field program. Always uses non-blocking sending and blocking
// receiving
void master(int numtasks) {
  int i = 0;

  int  rounds = 0;
  int *buffer = (int *)malloc((sizeof(int) * 3));

  printf("Initiating Master Field Program..\n");

  // Initialize every player, send them their info
  for (i = 1; i < numtasks; i++) {}

  while (rounds++ < 2) {}
}

// Player program. Always using blocking sends and receives
void player(int rank) {
  int type, posX, posY, ballX, ballY;
  MPI_Status stat;

  int *buffer = (int *)malloc((sizeof(int) * 3));

  do {
    MPI_RECV(buffer, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
    type = buffer[0];

    if (type == GAME_OVER) {
      continue;
    } else if (type == GAME_START) {
      posX = buffer[1];
      posY = buffer[2];
      printf("Player %i Reporting In! I'm at %i, %i\n", rank, posX, posY);
    } else if (type == ROUND_START) {
      ballX = buffer[1];
      ballY = buffer[2];
    } else if (type == KICK_PHASE) {}
    ballX = 0;
    ballY = 0;
  } while (type != GAME_OVER);
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

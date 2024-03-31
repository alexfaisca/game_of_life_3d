#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define N_SPECIES 9

/* gcc -ansi -pedantic -fopenmp life3d.c */
typedef struct {
  int64_t c;
  int64_t g;
} Statistics;
Statistics stats[N_SPECIES] = {0};
unsigned int seed;
char ***grid;

void print_g(int64_t gen, int64_t n, char **g) {
  int64_t x, y;
  printf("Layer %ld:\n", x);
  for (x = 0; x < n; x++) {
    for (y = 0; y < n; y++) {
      if (g[x][y] != 0)
        printf("%d ", g[x][y]);
      else
        printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}

void init_r4uni(int input_seed) { seed = input_seed + 987654321; }

float r4_uni() {
  int seed_in = seed;

  seed ^= (seed << 13), seed ^= (seed >> 17), seed ^= (seed << 5);

  return 0.5 + 0.2328306e-09 * (seed_in + (int)seed);
}

char ***gen_initial_grid(int64_t n, float density, int input_seed,
                         int world_size, int rank) {
  int64_t x, y, z, j, actual_size;
  actual_size = n / world_size + (rank < n % world_size) + 4;
  if (!(grid = malloc((actual_size) * sizeof(char **)))) {
    printf("Failed to allocate matrix\n");
    exit(1);
  }
#ifdef DEBUG
  printf("RANK %d\n", rank);
#endif
  if (rank != 0)
    MPI_Recv(&seed, 1, MPI_UNSIGNED, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (x = 0; x < actual_size; x++) {
    if (!(grid[x] = malloc(n * sizeof(char *)))) {
      printf("Failed to allocate matrix\n");
      exit(1);
    }
    if (!(grid[x][0] = calloc(n * n, sizeof(char)))) {
      printf("Failed to allocate matrix\n");
      exit(1);
    }
    for (y = 1; y < n; y++)
      grid[x][y] = grid[x][0] + y * n;
  }

  if (rank == 0)
    init_r4uni(input_seed);
  for (x = 1; x <= n / world_size + (n % world_size > rank); x++)
    for (y = 0; y < n; y++)
      for (z = 0; z < n; z++)
        if (r4_uni() < density)
          grid[x][y][z] = (int)(r4_uni() * N_SPECIES) + 1,
          stats[grid[x][y][z] - 1].c++;

  if (rank != world_size - 1 && world_size > 1)
    MPI_Send(&seed, 1, MPI_UNSIGNED, rank + 1, 0, MPI_COMM_WORLD);

  return grid;
}

void destroy_grid(int64_t n, int world_size, int rank) {
  n = world_size > 1 ? n / world_size + (rank < n % world_size) + 4 : n + 4;
  for (; n > 0; free(grid[--n][0]), free(grid[n]))
    ;
  free(grid);
}

void synchronize_extremities(int64_t n, int world_size, int rank) {
  int64_t t = n * n;
  char ***last_row;
  last_row = grid + n / world_size + (rank < n % world_size);
  if (rank == 0) {
    MPI_Send(&(last_row[0][0][0]), t, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&(last_row[1][0][0]), t, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (world_size % 2) {
      MPI_Send(&(grid[1][0][0]), t, MPI_BYTE, world_size - 1, 0, MPI_COMM_WORLD);
      MPI_Recv(&(grid[0][0][0]), t, MPI_BYTE, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(&(grid[0][0][0]), t, MPI_BYTE, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&(grid[1][0][0]), t, MPI_BYTE, world_size - 1, 0, MPI_COMM_WORLD);
    }
  } else if (rank == world_size - 1) {
    if (world_size % 2) {
      MPI_Send(&(grid[1][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD);
      MPI_Recv(&(grid[0][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&(last_row[1][0][0]), t, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&(last_row[0][0][0]), t, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&(grid[0][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&(grid[1][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD);
      MPI_Send(&(last_row[0][0][0]), t, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(&(last_row[1][0][0]), t, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  } else if (rank % 2) {
    MPI_Recv(&(grid[0][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&(grid[1][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&(last_row[1][0][0]), t, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&(last_row[0][0][0]), t, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Send(&(last_row[0][0][0]), t, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&(last_row[1][0][0]), t, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&(grid[1][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&(grid[0][0][0]), t, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void singleton_extremeties(int64_t n) {
  memcpy(&(grid[0][0][0]), &(grid[n][0][0]), n * n * sizeof(char));
  memcpy(&(grid[n + 1][0][0]), &(grid[1][0][0]), n * n * sizeof(char));
}

void print_grid(int64_t gen, int64_t n) {
  int64_t x, y, z;
  printf("Generation %ld ------------------------------\n", gen);
  for (x = 0; x < n + 2; x++) {
    printf("Layer %ld:\n", x);
    for (y = 0; y < n; y++) {
      for (z = 0; z < n; z++) {
        if (grid[x][y][z] != 0) printf("%d ", grid[x][y][z]);
        else printf("  ");
      }
      printf("\n");
    }
    printf("\n");
  }
}

int get_neighbours(int64_t n, int64_t x, int64_t y, int64_t z,
                   int64_t nb[N_SPECIES + 1]) {
  int i, j, k, v = grid[x][y][z], c = v > 0 ? -1 : 0;
  int64_t y_s[3] = {(n + y - 1) % n, y, (n + y + 1) % n},
          x_s[3] = {x - 1, x, x + 1},
          z_s[3] = {(n + z - 1) % n, z, (n + z + 1) % n};

  for (nb[v]--, i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
        if ((v = grid[x_s[i]][y_s[j]][z_s[k]]) != 0)
          c++, nb[v]++;
  return c;
}

void simulation_synchronized(int64_t gen, int64_t n, int world_size, int rank) {
  char **swap_space;
  int64_t x, y, z, c, s, i, g, xx, neighbours;
  int64_t total[N_SPECIES] = {0}, count[N_SPECIES + 1] = {0};
  xx = world_size > 1 ? n / world_size + (rank < n % world_size) : n;
  if (world_size > 1) {
#ifdef DEBUG
    printf("Before stat sync: ");
    for (x = 0; x < N_SPECIES; x++) printf("%ld ", stats[x].c);
    printf("\n");
#endif
    for (x = 0; x < N_SPECIES; total[x] = stats[x].c, x++)
      ;
    MPI_Reduce(total, count, N_SPECIES, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    printf("After stat sync: ");
#endif
    for (x = 0; x < N_SPECIES; total[x] = count[x] = 0, x++)
      stats[x] = (Statistics){count[x], 0}
#ifdef DEBUG
      , printf("%ld ", stats[x].c);
    printf("\n");
#else
      ;
#endif
  }
  for (g = 0;; g++) {
    if (world_size > 1) synchronize_extremities(n, world_size, rank);
    else singleton_extremeties(n);
#ifdef DEBUG
    print_grid(g, n);
#endif
    if (g >= gen)
      break;
    for (x = 1; x <= xx; x++) {
      for (y = 0; y < n; y++) {
        for (z = 0; z < n; z++) {
          neighbours = get_neighbours(n, x, y, z, count);
          for (s = 1, i = 1, c = 0; s <= N_SPECIES; count[s++] = 0)
            if (count[s] > c) i = s, c = count[s];

          if (grid[x][y][z] == 0) {
            if (neighbours > 6 && neighbours < 11) 
	      total[(grid[xx + 3][y][z] = i) - 1]++;
            else grid[xx + 3][y][z] = 0;
          } else if (neighbours < 5 || neighbours > 13) grid[xx + 3][y][z] = 0;
          else total[(grid[xx + 3][y][z] = grid[x][y][z]) - 1]++;
        }
      }
      if (x > 1) {
        swap_space = grid[x - 1];
        grid[x - 1] = grid[xx + 2];
        grid[xx + 2] = grid[xx + 3];
        grid[xx + 3] = swap_space;
        if (x == xx) { // case x == n -1
          swap_space = grid[x];
          grid[x] = grid[xx + 2];
          grid[xx + 2] = swap_space;
        }
      } else { // case x == 1
        if (x == xx) {
          swap_space = grid[x];
	  grid[x] = grid[xx + 3];
	  grid[xx + 3] = swap_space;
        } else {
          swap_space = grid[xx + 2], grid[xx + 2] = grid[xx + 3]; 
	  grid[xx + 3] = swap_space;
        }
      }
    }

    if (world_size > 1) {
      MPI_Reduce(total, count, N_SPECIES, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
      for (x = 0; x < N_SPECIES; total[x] = count[x] = 0, x++)
        if (count[x] > stats[x].c && rank == 0)
          stats[x] = (Statistics){count[x], g + 1};
    } else for (x = 0; x < N_SPECIES; total[x] = 0, x++)
        if (total[x] > stats[x].c) stats[x] = (Statistics){total[x], g + 1};
  }
  return;
}

void print_result(void) {
  int i;
  for (i = 0; i < N_SPECIES; i++)
    printf("%d %ld %ld\n", i + 1, stats[i].c, stats[i].g);
  return;
}

int main(int argc, char *argv[]) {
  int world_size, world_rank, d;
  long gen = atol(argv[1]), n = atol(argv[2]);
  double exec_time, density = atof(argv[3]);
  seed = atoi(argv[4]);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  grid = gen_initial_grid(n, density, seed, world_size, world_rank);
  if (world_size == 1 || world_rank == 0) exec_time = -omp_get_wtime();

  simulation_synchronized(gen, n, world_size, world_rank);

  if (world_size == 1 || world_rank == 0)
    exec_time += omp_get_wtime(), print_result(),
        fprintf(stderr, "%.1fs\n", exec_time);
  destroy_grid(n, world_size, world_rank);

  MPI_Barrier(MPI_COMM_WORLD), MPI_Finalize();
  return 0;
}

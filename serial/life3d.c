/* DISCLAIMER */
/* Made exclusively by Alexandre Coelho, student number 100120 */
/* END OF DISCLAIMER */

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

char ***gen_initial_grid(int N, float density, int input_seed) {
  int x, y, z;

  if ((grid = malloc((N + 3) * sizeof(char **))) == NULL) {
    printf("Failed to allocate matrix\n");
    exit(1);
  }
  for (x = 0; x < N + 3; x++) {
    if ((grid[x] = malloc(N * sizeof(char *))) == NULL) {
      printf("Failed to allocate matrix\n");
      exit(1);
    }
    if ((grid[x][0] = calloc(N * N, sizeof(char))) == NULL) {
      printf("Failed to allocate matrix\n");
      exit(1);
    }
    for (y = 1; y < N; y++)
      grid[x][y] = grid[x][0] + y * N;
  }

  init_r4uni(input_seed);
  for (x = 0; x < N; x++)
    for (y = 0; y < N; y++)
      for (z = 0; z < N; z++)
        if (r4_uni() < density)
          grid[x][y][z] = (int)(r4_uni() * N_SPECIES) + 1,
          stats[grid[x][y][z] - 1].c++;
  return grid;
}

void destroy_grid(int n) {
  for (; n > 0; free(grid[--n][0]), free(grid[n]))
    ;
  free(grid);
}

void print_grid(int64_t gen, int64_t n) {
  int64_t x, y, z;
  printf("Generation %ld ------------------------------\n", gen);
  for (x = 0; x < n; x++) {
    printf("Layer %ld:\n", x);
    for (y = 0; y < n; y++) {
      for (z = 0; z < n; z++) {
        if (grid[x][y][z] != 0)
          printf("%d ", grid[x][y][z]);
        else
          printf("  ");
      }
      printf("\n");
    }
    printf("\n");
  }
}

int get_neighbours(int64_t n, int64_t x, int64_t y, int64_t z,
                   int nb[N_SPECIES + 1]) {
  int v = grid[x][y][z];
  int a, i, j, k, c = v > 0 ? -1 : 0;
  int64_t y_s[] = {(n + y - 1) % n, y, (n + y + 1) % n},
          x_s[] = {(n + x - 1) % n, x, (n + x + 1) % n},
          z_s[] = {(n + z - 1) % n, z, (n + z + 1) % n};

  for (nb[v]--, i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
        if ((v = grid[x_s[i]][y_s[j]][z_s[k]]) != 0)
          c++, nb[v]++;
  return c;
}

void simulation(int64_t gen, int64_t n) {
  char **swap_space;
  int *count, neighbours, c, s, i;
  int64_t total[N_SPECIES] = {0}, g, xx, x, yy, y, zz, z;
  if ((count = calloc((N_SPECIES + 1), sizeof(int))) == NULL) {
    printf("Failed to allocate matrix\n");
    exit(1);
  }
  for (g = 0;; g++) {
    /* DEGUB */
    // print_grid(g, n);
    if (g >= gen)
      break;

    for (x = 0; x < n; x++) {
      for (y = 0; y < n; y++) {
        for (z = 0; z < n; z++) {
          neighbours = get_neighbours(n, x, y, z, count);
          for (s = 1, i = 1, c = 0; s <= N_SPECIES; count[s++] = 0)
            if (count[s] > c)
              i = s, c = count[s];

          if (grid[x][y][z] == 0) {
            if (neighbours > 6 && neighbours < 11)
              total[(grid[n + 1][y][z] = i) - 1]++;
            else
              grid[n + 1][y][z] = 0;
          } else if (neighbours < 5 || neighbours > 13) {
            grid[n + 1][y][z] = 0;
          } else {
            total[(grid[n + 1][y][z] = grid[x][y][z]) - 1]++;
          }
        }
      }
      // Efectuate results
      if (x > 1) {
        swap_space = grid[x - 1];
        grid[x - 1] = grid[n];
        grid[n] = grid[n + 1];
        grid[n + 1] = swap_space;
        if (x == n - 1) { // case x == n -1
          swap_space = grid[x];
          grid[x] = grid[n];
          grid[n] = grid[0];
          grid[0] = grid[n + 2];
          grid[n + 2] = swap_space;
        }
      } else { // case x == 0 or x == 1
        if (x == 1) {
          swap_space = grid[n];
          grid[n] = grid[n + 1];
          grid[n + 1] = swap_space;
        } else {
          swap_space = grid[n + 2];
          grid[n + 2] = grid[n + 1];
          grid[n + 1] = swap_space;
        }
      }
    }
    for (x = 0; x < N_SPECIES; total[x] = 0, x++)
      if (total[x] > stats[x].c)
        stats[x] = (Statistics){total[x], g + 1};
  }
  free(count);
  return;
}

void print_result(void) {
  int i;
  for (i = 0; i < N_SPECIES; i++)
    printf("%d %ld %ld\n", i + 1, stats[i].c, stats[i].g);
  return;
}

int main(int argc, char *argv[]) {
  long gen = atol(argv[1]), n = atol(argv[2]);
  double exec_time, density = atof(argv[3]);
  long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);

  seed = atoi(argv[4]);
  grid = gen_initial_grid(n, density, seed);
  exec_time = -omp_get_wtime();
  simulation(gen, n);
  exec_time += omp_get_wtime();
  print_result();
  // fprintf(stderr, "%.1fs\n", exec_time);
  destroy_grid(n + 3);
  return 0;
}

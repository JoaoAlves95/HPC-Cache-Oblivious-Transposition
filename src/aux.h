#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <xmmintrin.h>

#define SZ(l) ((1<<((l)<<1)))              //(4^l)-1
#define POW2(u) ((unsigned int) 1<< (u))
#define LOG2(u) (((int32_t)31 - (int32_t)__builtin_clz((u))))
#define GROUP(n) ((n/4) + ((n % 4) != 0))

void print_curve(const uint8_t* c, int sz){
  for(int i = 0; i < sz; i++)
    printf("%d ",c[i]);
  printf("\n");
}


float* zero_matrix(int n, int m)
{
	float *M = _mm_malloc(sizeof(*M) * (n * m), 64);
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			M[(i*m) + j] = 0.0f;
	return M;
}

float* float_matrix(int n, int m)
{
	float *M = _mm_malloc(sizeof(*M) * (n * m),64);
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			M[(i*m) + j] = (float) i;
		}
	}
	return M;
}

float* rndm_matrix(int n, int m)
{
	float *M = _mm_malloc(sizeof(*M) * (n * m),64);
	
	time_t t;
   	srand((unsigned) time(&t));

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			M[(i*m) + j] = (float) (rand() % 50);
		}
	}
	return M;
}



void printmatrix(float *M, int n, int m)
{
	int c=0;
	printf("Printing Matrix:\n");
	for(int i=0; i<n ;i++)
	{
        printf("l%d\t\t",i);
		for(int j=0; j<m; j++){
			printf("%d ",(int) M[(i*m)+j]);
			c++;
		}
		printf("\n");
	}
	printf("c:%d\n",c);
}

void print_result(float time, char* fname, int i, int j)
{
    char str[30];
    sprintf(str, "data/%s:%dx%d",fname,i, j);
    FILE *f = fopen(str,"a");
    if(f != NULL)
        fprintf(f,"%f\n", time);
    fclose(f);
}

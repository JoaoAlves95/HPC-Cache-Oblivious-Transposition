#include "aux.h"

unsigned int counter = 0;

void transpose4x4_SSE(const float *A, float *B, const int lda, const int ldb)
{
	__m128 row1 = _mm_load_ps(&A[0*lda]);
	__m128 row2 = _mm_load_ps(&A[1*lda]);
	__m128 row3 = _mm_load_ps(&A[2*lda]);
	__m128 row4 = _mm_load_ps(&A[3*lda]);
	_MM_TRANSPOSE4_PS(row1, row2, row3, row4);
	_mm_store_ps(&B[0*ldb], row1);
	_mm_store_ps(&B[1*ldb], row2);
	_mm_store_ps(&B[2*ldb], row3);
	_mm_store_ps(&B[3*ldb], row4);
}

void oop_transpose(const float *src, float *dst, const unsigned int src_stride, const unsigned int dst_stride)
{
	unsigned int n_delta = dst_stride >> 2;
	unsigned int m_delta = src_stride >> 2;
	unsigned int n_level = LOG2(n_delta);
	unsigned int m_level = LOG2(m_delta);

	const unsigned int level = (n_level > m_level) ? m_level-1 : n_level-1;

	unsigned int a,v,d,n,m,c;
	if(n_level < m_level)
	{
		a = 1;
		v = 0;
		d = (level % 2) ? 3 : 2;
		n = n_delta;
		m = POW2(LOG2(n_delta)) + (m_delta % POW2(LOG2(n_delta)));
		c = m_delta / POW2(LOG2(n_delta));
	}
	else if(n_level > m_level)
	{
		a = 1;
		v = 1;
		d = (level % 2) ? 2 : 3;
		m = m_delta;
		n = POW2(LOG2(m_delta)) + (n_delta % POW2(LOG2(m_delta)));
		c = n_delta / POW2(LOG2(m_delta));
	}
	else 
	{
		a = 0;
		v = 0;
		d = 3;
		n = n_delta;
		m = m_delta;
		c = 1;
	}

	//draw curve here
	uint8_t s1 = (d == 3) ? 2 : 3;
	uint8_t s2 = (d == 3) ? 3 : 2;
	uint8_t s3 = (d == 3) ? 0 : 1;

	//Alloc Curve 
	uint8_t* curve = _mm_malloc(sizeof(*curve)*((SZ(level)<16)?16:SZ(level)), 64);
	//Init Curve 
	if(d==3)
	{
		curve[0] = 3;curve[1] = 2;curve[2] = 1;curve[3] = 2;
		curve[4] = 2;curve[5] = 3;curve[6] = 0;curve[7] = 3;
		curve[8] = 2;curve[9] = 3;curve[10] = 0;curve[11] = 0;
		curve[12] = 1;curve[13] = 0;curve[14] = 3;curve[15] = 0;
	}
	else
	{
		curve[0] = 2;curve[1] = 3;curve[2] = 0;curve[3] = 3;
		curve[4] = 3;curve[5] = 2;curve[6] = 1;curve[7] = 2;
		curve[8] = 3;curve[9] = 2;curve[10] = 1;curve[11] = 1;
		curve[12] = 0;curve[13] = 1;curve[14] = 2;curve[15] = 0;
	}

	for(int l = 2; l < level; l++)
    { 
        const unsigned int sz = SZ(l);
        const unsigned int offset_q2 = sz;
        const unsigned int offset_q3 = sz * 2;
        const unsigned int offset_q4 = sz * 3;

        const __m128i xor1 = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        const __m128i xor2 = _mm_set_epi8(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
            
        for(int c=0; c < sz; c+=16)
        {
                __m128i q1 = _mm_loadu_si128((__m128i *) &curve[c]);
                _mm_storeu_si128 ((__m128i*) &curve[offset_q2 + c], _mm_xor_si128(q1 , xor1));
                _mm_storeu_si128 ((__m128i*) &curve[offset_q3 + c], _mm_xor_si128(q1 , xor1));
                _mm_storeu_si128 ((__m128i*) &curve[offset_q4 + c], _mm_xor_si128(q1 , xor2));
        }

        curve[offset_q2-1] = (s1^=1);
        curve[offset_q3-1] = (s2^=1);
        curve[offset_q4-1] = (s3^=1);
    }

	const uint8_t *hilbert_curve = curve;

	unsigned int J_floor = 0;
	unsigned int I_floor = 0;
	for(int r = 0; r < c; r++)
	{
		const unsigned int dim = (n < m) ? n : m;
		const unsigned int bot = POW2(LOG2(dim));
		const unsigned int mid = bot + (bot >> 1);
		const unsigned int num = bot >> 1;

		unsigned int n_blk_0, n_blk_1, m_blk_0, m_blk_1, tmp_n_blk_type, tmp_m_blk_type;

		if(n <= mid)
		{
			n_blk_1 = n - bot;
			n_blk_0 = num - n_blk_1;
			tmp_n_blk_type = 0;
		}else
		{
			n_blk_1 = n - mid;
			n_blk_0 = num - n_blk_1;
			tmp_n_blk_type = 1;
		}

		if(m <= mid)
		{
			m_blk_1 = m - bot;
			m_blk_0 = num - m_blk_1;
			tmp_m_blk_type = 0;
		}else
		{
			m_blk_1 = m - mid;
			m_blk_0 = num - m_blk_1;
			tmp_m_blk_type = 1;
		}
	
		const unsigned int n_blk[2] = {n_blk_0, n_blk_1};
		const unsigned int m_blk[2] = {m_blk_0, m_blk_1};
		const unsigned int n_blk_type = tmp_n_blk_type;
		const unsigned int m_blk_type = tmp_m_blk_type;

		//auxiliary block variables
		const unsigned int blk_values[2][2] = {{2,3},{3,4}};
		const unsigned int n_mid = n_blk[0];
		const unsigned int m_mid = m_blk[0];

		//block loop variables
		unsigned int J_bot = J_floor;
		unsigned int I_bot = I_floor;
		unsigned int J_top = J_bot + blk_values[m_blk_type][(0 < m_mid) ? 0 : 1];
		unsigned int I_top = I_bot + blk_values[n_blk_type][(0 < n_mid) ? 0 : 1];
		
		unsigned int i=0,j=0, H = num * num;
		for(int h = 0; h < H; h++)
		{	
			unsigned int aux_J_bot = J_bot;
			unsigned int aux_J_top = J_top;
			unsigned int aux_I_bot = I_bot;
			unsigned int aux_I_top = I_top;

			uint8_t d = hilbert_curve[h];
			int J_sign = (d-1) % 2;
			int I_sign = (d-2) % 2;

			if(J_sign < 0){
				j--;
				J_bot -= blk_values[m_blk_type][(j < m_mid) ? 0 : 1];
				J_top -= blk_values[m_blk_type][(j+1 < m_mid) ? 0 : 1];
			}
			else if(J_sign > 0){
				J_bot += blk_values[m_blk_type][(j < m_mid) ? 0 : 1];
				J_top += blk_values[m_blk_type][(j+1 < m_mid) ? 0 : 1];
				j++;
			}
			else if(I_sign < 0){
				i--;
				I_bot -= blk_values[n_blk_type][(i < n_mid) ? 0 : 1];
				I_top -= blk_values[n_blk_type][(i+1 < n_mid) ? 0 : 1];
			}
			
			else if(I_sign > 0){
				I_bot += blk_values[n_blk_type][(i < n_mid) ? 0 : 1];
				I_top += blk_values[n_blk_type][(i+1 < n_mid) ? 0 : 1];
				i++;
			}

			int k_max;
			unsigned int i_idx = I_bot * 4;
			unsigned int j_idx = J_bot * 4;

			//Matrix src prefetch
			k_max = 4 * (I_top - I_bot);
			for(int k=0; k < k_max; k++)
				__builtin_prefetch(&src[((i_idx + k) * src_stride)+j_idx],0,3);

			//Matrix dst prefetch
			k_max = 4 * (J_top - J_bot);
			for(int k=0; k < k_max; k++)
				__builtin_prefetch(&dst[((j_idx + k) * dst_stride) + i_idx],1,3);

			for(unsigned int y = aux_I_bot; y < aux_I_top; y++)
			{
				for(unsigned int x = aux_J_bot; x < aux_J_top; x++)
				{
					unsigned int Y = y * 4;
					unsigned int X = x * 4;
					transpose4x4_SSE(&src[(Y*src_stride)+X], &dst[(X*dst_stride)+Y], src_stride, dst_stride);
				}
			}
		}
		//set values for next assymetrical case
		if(a)
		{
			if(v)
			{
				n = POW2(LOG2(m_delta));
				I_floor += (n_blk[0] * blk_values[n_blk_type][0]) + (n_blk[1] * blk_values[n_blk_type][1]);
				J_floor = 0;
			} 
			else
			{
				m = POW2(LOG2(n_delta));
				J_floor += (m_blk[0] * blk_values[m_blk_type][0]) + (m_blk[1] * blk_values[m_blk_type][1]);
				I_floor = 0;
			}
		}
	}
	_mm_free(curve);
}

int main(int argc, char** argv){
	unsigned int dst_stride = atoi(argv[1]);
	unsigned int src_stride = atoi(argv[2]);
	float* M = rndm_matrix(dst_stride,src_stride);
	float* T = zero_matrix(src_stride,dst_stride);

  clock_t t = clock();
	oop_transpose(M, T, src_stride,dst_stride);
  
  t = clock() - t;
	double sec = ((double)t)/CLOCKS_PER_SEC;
	printf("Transpose time: %f\n",sec);
  
	time_t tmp;
	srand((unsigned) time(&tmp));
	int n1 = rand()%dst_stride;
	int m1 = rand()%src_stride;
 	printf("%f\n", T[(n1 * m1)-1]);

	_mm_free(M);
	_mm_free(T);
	return 0;
}

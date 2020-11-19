#include <stdio.h>
#include <string.h>
#include <immintrin.h>

const char* dgemm_desc = "Single thread dgemm";

#define MIN(a,b) (((a)<(b))?(a):(b))

#define KC 256
#define MC 96
#define MR 8
#define NR 6

#define BUFSZA (KC * MC)
#define BUFSZB (KC * NR)
#define BUFSZC (MR * NR)

void mma864(const double *a, int lda, const double *b, int ldb, double *c, int ldc) {
    __m256d frag_c[2][6];
    __m256d frag_a[2];
    __m256d frag_b[2];

    double buf[4];

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            frag_c[i][j] = _mm256_load_pd(&c[i * 4 + j * ldc]);
        }
    }
   	
    for (int p = 0; p < 4; p++) {
        for (int i = 0; i < 2; i++) {
            frag_a[i] = _mm256_load_pd(&a[i * 4 + p * lda]);
        }

        for (int t = 0; t < 3; t++) {
            for (int i = 0; i < 2; i++) {
                frag_b[i] = _mm256_broadcast_sd(&b[p*ldb+2*t+i]);
            }
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    frag_c[i][2*t+j] = _mm256_fmadd_pd(frag_a[i], frag_b[j], frag_c[i][2*t+j]);
                }
            }
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            _mm256_storeu_pd(&c[i * 4 + j * ldc], frag_c[i][j]);
        }
    }
}

void square_dgemm (int lda, const double* A, const double* B, double* C) {
    double buf_a[BUFSZA] __attribute__((aligned(64)));
    double buf_b[BUFSZB] __attribute__((aligned(64)));
    double buf_c[BUFSZC] __attribute__((aligned(64)));

    for (int k1 = 0; k1 < (lda-1)/KC+1; k1++) {
        for (int i1 = 0; i1 < (lda - 1) / MC + 1; i1++) {
            // pcak A buffer MC * KC
            memset(buf_a, 0, sizeof(buf_a));
            for (int k = 0; k < MIN(KC, lda-k1*KC); k++) {
                for (int i = 0; i < MIN(MC, lda-i1*MC); i++) {
                    buf_a[(i/MR)*MR*KC + k * MR + i%MR] = A[i1*MC+i+(k1*KC+k)*lda];
                }
            }

            for (int j = 0; j < lda; j += NR) {
                // pack B buffer KC * NR
                memset(buf_b, 0, sizeof(buf_b));
                for (int k = 0; k < MIN(KC, lda-k1*KC); k++) {
                    for (int j0 = 0; j0 < MIN(NR, lda-j); j0++) {
                        buf_b[k * NR + j0] = B[(k1*KC+k) + (j + j0) * lda];
                    }
                }

                for (int i = 0; i < MIN(MC, lda-i1*MC); i += MR) {
                    // Matmul 64 * 256 * 8
                    for (int j0 = 0; j0 < MIN(NR,lda-j); j0++) {
                        for (int i0 = 0; i0 < MIN(MR,lda-i1*MC-i); i0++) {
                            buf_c[i0 + j0 * MR] = C[i1 * MC + i + i0 + (j + j0) * lda];
                        }
                    }
                    for (int k = 0; k < MIN(KC,lda-k1*KC); k += 4) {
                        mma864(buf_a + i * KC + k * MR, MR, buf_b + k * NR, NR, buf_c, MR);
                    }
                    for (int j0 = 0; j0 < MIN(NR,lda-j); j0++) {
                        for (int i0 = 0; i0 < MIN(MR,lda-i1*MC-i); i0++) {
                            C[i1 * MC + i + i0 + (j + j0) * lda] = buf_c[i0 + j0 * MR];
                        }
                    }
                }
            }
        }
    }
}

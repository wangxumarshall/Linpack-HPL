/*
 * HPL_sdc_checksum.c - Core checksum computation routines for SDC detection
 */
#include "hpl.h"

#ifdef HPL_SDC_CHECK

#ifdef STDC_HEADERS
void HPL_sdc_init_weights
(
   double * w,
   int      n
)
#else
void HPL_sdc_init_weights( w, n )
   double * w; int n;
#endif
{
/*
 * Purpose
 * =======
 * Initialize weight vector with power-of-two weights using modular
 * window to prevent overflow. w[i] = 2^(i % HPL_SDC_WEIGHT_WINDOW)
 */
   int i;
   for( i = 0; i < n; i++ )
   {
      w[i] = (double)( 1 << ( i % HPL_SDC_WEIGHT_WINDOW ) );
   }
}

#ifdef STDC_HEADERS
double HPL_sdc_col_checksum
(
   const double * A,
   int            lda,
   int            m,
   int            n,
   const double * weights
)
#else
double HPL_sdc_col_checksum( A, lda, m, n, weights )
   const double * A; int lda, m, n; const double * weights;
#endif
{
/*
 * Purpose
 * =======
 * Compute weighted column checksum: cs = sum_j (sum_i w[i] * A[i][j])
 */
   double cs = 0.0;
   int i, j;

   for( j = 0; j < n; j++ )
   {
      double col_sum = 0.0;
      for( i = 0; i < m; i++ )
      {
         col_sum += weights[i] * A[i + j * lda];
      }
      cs += col_sum;
   }
   return cs;
}

#ifdef STDC_HEADERS
void HPL_sdc_panel_checksum
(
   const double * A,
   int            lda,
   int            m,
   int            n,
   const double * weights,
   double       * cs_out
)
#else
void HPL_sdc_panel_checksum( A, lda, m, n, weights, cs_out )
   const double * A; int lda, m, n;
   const double * weights; double * cs_out;
#endif
{
/*
 * Purpose
 * =======
 * Compute per-column checksums for a panel: cs[k] = sum_i w[i]*A[i][k]
 */
   int i, k;

   for( k = 0; k < n; k++ )
   {
      double s = 0.0;
      for( i = 0; i < m; i++ )
      {
         s += weights[i] * A[i + k * lda];
      }
      cs_out[k] = s;
   }
}

#ifdef STDC_HEADERS
void HPL_sdc_update_trail_checksum
(
   double       * cs_trail,
   const double * L2,
   int            ldl2,
   const double * U,
   int            ldu,
   int            mp,
   int            jb,
   int            nn,
   const double * weights
)
#else
void HPL_sdc_update_trail_checksum( cs_trail, L2, ldl2, U, ldu,
                                     mp, jb, nn, weights )
   double * cs_trail; const double * L2; int ldl2;
   const double * U; int ldu, mp, jb, nn; const double * weights;
#endif
{
/*
 * Purpose
 * =======
 * Incremental checksum update for trailing matrix update A -= L2 * U.
 *
 * cs_trail[j] -= sum_k cs_L2[k] * U[k][j]
 * where cs_L2[k] = sum_i w[i] * L2[i][k]
 *
 * Cost: O(mp*jb + jb*nn) vs main dgemm O(mp*jb*nn)
 * Overhead ratio: ~1/min(mp,nn) << 1
 */
   int i, j, k;
   double cs_L2[512];  /* stack buffer; jb typically <= 512 */
   double * cs_l2_ptr;
   int    alloc = 0;

   if( jb > 512 )
   {
      cs_l2_ptr = (double *)malloc( (size_t)jb * sizeof(double) );
      alloc = 1;
   }
   else
   {
      cs_l2_ptr = cs_L2;
   }

   /* Compute checksum of each L2 column */
   for( k = 0; k < jb; k++ )
   {
      double s = 0.0;
      for( i = 0; i < mp; i++ )
      {
         s += weights[i] * L2[i + k * ldl2];
      }
      cs_l2_ptr[k] = s;
   }

   /* Update trailing checksum: cs_trail[j] -= sum_k cs_L2[k] * U[k][j] */
   for( j = 0; j < nn; j++ )
   {
      double upd = 0.0;
      for( k = 0; k < jb; k++ )
      {
         upd += cs_l2_ptr[k] * U[k + j * ldu];
      }
      cs_trail[j] -= upd;
   }

   if( alloc ) free( cs_l2_ptr );
}

#ifdef STDC_HEADERS
void HPL_sdc_compute_bcast_checksum
(
   const double * L2,
   int            ldl2,
   int            ml2,
   const double * L1,
   int            jb_l1,
   const double * DPIV,
   int            jb,
   double       * cs_out
)
#else
void HPL_sdc_compute_bcast_checksum( L2, ldl2, ml2, L1, jb_l1, DPIV, jb, cs_out )
   const double * L2; int ldl2, ml2; const double * L1; int jb_l1;
   const double * DPIV; int jb; double * cs_out;
#endif
{
/*
 * Purpose
 * =======
 * Compute checksum of the broadcast buffer (L2 + L1 + DPIV).
 *
 * L2 holds ml2 rows x jb cols of the panel, laid out with leading
 * dimension ldl2 (ldl2 >= ml2). We sum only the ml2*jb real panel
 * entries, NOT the stride padding, so the checksum is independent of
 * the memory layout and matches across processes with different ldl2.
 */
   double cs = 0.0;
   int i, k;

   /* Checksum L2 (sum real panel entries only) */
   for( k = 0; k < jb; k++ )
      for( i = 0; i < ml2; i++ )
         cs += L2[i + k * ldl2];

   /* Checksum L1 */
   for( i = 0; i < jb_l1 * jb_l1; i++ )
      cs += L1[i];

   /* Checksum DPIV */
   for( i = 0; i < jb; i++ )
      cs += DPIV[i];

   *cs_out = cs;
}

#endif /* HPL_SDC_CHECK */

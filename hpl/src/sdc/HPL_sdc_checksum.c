/*
 * HPL_sdc_checksum.c - Core checksum computation routines for SDC detection
 */
#include "hpl.h"

#ifdef HPL_SDC_CHECK

#ifdef STDC_HEADERS
void HPL_sdc_init_weights
(
   double * w,
   const int n
)
#else
void HPL_sdc_init_weights( w, n )
   double * w; const int n;
#endif
{
/*
 * Purpose
 * =======
 * Initialize weight vector with power-of-two weights using modular
 * window to prevent overflow. w[i] = 2^(i % HPL_SDC_WEIGHT_WINDOW)
 */
   int i;
   if( !w || n <= 0 ) return;
   for( i = 0; i < n; i++ )
   {
      w[i] = (double)( 1ULL << ( (unsigned int)i % HPL_SDC_WEIGHT_WINDOW ) );
   }
}

#ifdef STDC_HEADERS
double HPL_sdc_col_checksum
(
   const double * A,
   const int      lda,
   const int      m,
   const int      n,
   const double * weights
)
#else
double HPL_sdc_col_checksum( A, lda, m, n, weights )
   const double * A; const int lda, m, n; const double * weights;
#endif
{
/*
 * Purpose
 * =======
 * Compute weighted column checksum with Kahan compensated summation:
 * cs = sum_j (sum_i w[i] * A[i][j])
 */
   double sum = 0.0, c = 0.0, y, t;
   int i, j;

   if( !A || lda < m || m <= 0 || n <= 0 ) return 0.0;

   for( j = 0; j < n; j++ )
   {
      for( i = 0; i < m; i++ )
      {
         double w = weights ? weights[i] : 1.0;
         y = w * A[i + (size_t)j * lda] - c;
         t = sum + y;
         c = ( t - sum ) - y;
         sum = t;
      }
   }
   return sum;
}

#ifdef STDC_HEADERS
void HPL_sdc_panel_checksum
(
   const double * A,
   const int      lda,
   const int      m,
   const int      n,
   const double * weights,
   double       * cs_out
)
#else
void HPL_sdc_panel_checksum( A, lda, m, n, weights, cs_out )
   const double * A; const int lda, m, n;
   const double * weights; double * cs_out;
#endif
{
/*
 * Purpose
 * =======
 * Compute per-column checksums for a panel using Kahan summation:
 * cs[k] = sum_i w[i]*A[i][k]
 */
   int i, k;

   if( !A || !cs_out || lda < m || m <= 0 || n <= 0 ) return;

   for( k = 0; k < n; k++ )
   {
      double sum = 0.0, c = 0.0, y, t;
      for( i = 0; i < m; i++ )
      {
         double w = weights ? weights[i] : 1.0;
         y = w * A[i + (size_t)k * lda] - c;
         t = sum + y;
         c = ( t - sum ) - y;
         sum = t;
      }
      cs_out[k] = sum;
   }
}



#ifdef STDC_HEADERS
void HPL_sdc_compute_bcast_checksum
(
   const double * L2,
   const int      ldl2,
   const int      ml2,
   const double * L1,
   const int      jb_l1,
   const double * DPIV,
   const int      jb,
   const double * weights,
   double       * cs_out
)
#else
void HPL_sdc_compute_bcast_checksum( L2, ldl2, ml2, L1, jb_l1, DPIV, jb, weights, cs_out )
   const double * L2; const int ldl2, ml2; const double * L1; const int jb_l1;
   const double * DPIV; const int jb; const double * weights; double * cs_out;
#endif
{
/*
 * Purpose
 * =======
 * Compute checksum of the broadcast buffer (L2 + L1 + DPIV).
 */
   double sum = 0.0, c = 0.0, y, t;
   int i, k;

   if( !cs_out ) return;

   /* Checksum L2 */
   if( L2 && ml2 > 0 && jb > 0 )
   {
      for( k = 0; k < jb; k++ )
      {
         for( i = 0; i < ml2; i++ )
         {
            double w = weights ? weights[i] : 1.0;
            y = w * L2[i + (size_t)k * ldl2] - c;
            t = sum + y;
            c = ( t - sum ) - y;
            sum = t;
         }
      }
   }

   /* Checksum L1 */
   if( L1 && jb_l1 > 0 )
   {
      for( i = 0; i < jb_l1 * jb_l1; i++ )
      {
         y = L1[i] - c;
         t = sum + y;
         c = ( t - sum ) - y;
         sum = t;
      }
   }

   /* Checksum DPIV */
   if( DPIV && jb > 0 )
   {
      for( i = 0; i < jb; i++ )
      {
         y = DPIV[i] - c;
         t = sum + y;
         c = ( t - sum ) - y;
         sum = t;
      }
   }

   *cs_out = sum;
}

#endif /* HPL_SDC_CHECK */

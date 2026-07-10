/*
 * HPL_sdc_verify.c - Verification routines for SDC detection
 */
#include "hpl.h"

#ifdef HPL_SDC_CHECK

#ifdef STDC_HEADERS
int HPL_sdc_verify_checksum
(
   const double cs_expected,
   const double cs_computed,
   const double threshold
)
#else
int HPL_sdc_verify_checksum( cs_expected, cs_computed, threshold )
   const double cs_expected, cs_computed, threshold;
#endif
{
/*
 * Purpose
 * =======
 * Compare two checksums with adaptive relative/absolute threshold.
 * Handles NaN and Inf properly.
 * Returns 0 if match, 1 if mismatch (SDC detected).
 */
   double denom, dev;

   if( isnan(cs_computed) || isinf(cs_computed) || isnan(cs_expected) || isinf(cs_expected) )
      return 1;

   dev = fabs( cs_computed - cs_expected );
   denom = fabs( cs_expected );

   if( denom < 1.0e-4 )
   {
      return ( dev > fmax(threshold, 1.0e-12) ) ? 1 : 0;
   }
   return ( ( dev / denom ) > threshold ) ? 1 : 0;
}



#ifdef STDC_HEADERS
int HPL_sdc_verify_panel_entry
(
   const double * A,
   const int      lda,
   const int      m,
   const int      n
)
#else
int HPL_sdc_verify_panel_entry( A, lda, m, n )
   const double * A; const int lda, m, n;
#endif
{
/*
 * Purpose
 * =======
 * Verify panel column block upon entry to factorization (JIT Check).
 * Checks SIMD/Cache-resident data for IEEE 754 NaN/Inf or numerical divergence
 * resulting from historical DGEMM trailing matrix updates.
 * Returns 1 if corruption/singularity detected, 0 otherwise.
 */
   int i, j;

   if( !A || m <= 0 || n <= 0 ) return 0;

   for( j = 0; j < n; j++ )
   {
      const double * col = A + (size_t)j * lda;
      for( i = 0; i < m; i++ )
      {
         double val = col[i];
         if( isnan(val) || isinf(val) || (val > 1.0e150) || (val < -1.0e150) )
         {
            return 1;
         }
      }
   }
   return 0;
}

#ifdef STDC_HEADERS
HPL_T_SDC_CONFIDENCE HPL_sdc_classify_panel_entry
(
   const double * A,
   const int      lda,
   const int      m,
   const int      n
)
#else
HPL_T_SDC_CONFIDENCE HPL_sdc_classify_panel_entry( A, lda, m, n )
   const double * A; const int lda, m, n;
#endif
{
/*
 * Purpose
 * =======
 * Classify a panel-entry anomaly by the strongest evidence observed.
 * NaN/Inf is confirmed; finite range overflow remains suspected.
 */
   int i, j;

   if( !A || m <= 0 || n <= 0 ) return HPL_SDC_SUSPECTED;

   for( j = 0; j < n; j++ )
   {
      const double * col = A + (size_t)j * lda;
      for( i = 0; i < m; i++ )
      {
         double val = col[i];
         if( isnan(val) || isinf(val) ) return HPL_SDC_CONFIRMED;
      }
   }
   return HPL_SDC_SUSPECTED;
}

#endif /* HPL_SDC_CHECK */

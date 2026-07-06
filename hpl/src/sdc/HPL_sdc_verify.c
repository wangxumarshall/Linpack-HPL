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
int HPL_sdc_verify_panel
(
   const double * A,
   const int      lda,
   const int      m,
   const int      n,
   const double * weights,
   const double * cs_expected,
   const double   threshold
)
#else
int HPL_sdc_verify_panel( A, lda, m, n, weights, cs_expected, threshold )
   const double * A; const int lda, m, n;
   const double * weights; const double * cs_expected; const double threshold;
#endif
{
/*
 * Purpose
 * =======
 * Verify panel checksums using robust Kahan summation from HPL_sdc_panel_checksum.
 * Returns number of mismatched columns (0 if all columns match).
 */
   int k, failed = 0;
   double cs_comp;

   if( !A || !cs_expected ) return 0;

   for( k = 0; k < n; k++ )
   {
      HPL_sdc_panel_checksum( A + (size_t)k * lda, lda, m, 1, weights, &cs_comp );
      if( HPL_sdc_verify_checksum( cs_expected[k], cs_comp, threshold ) )
         failed++;
   }
   return failed;
}

#ifdef STDC_HEADERS
int HPL_sdc_verify_trailing
(
   const double * A,
   const int      lda,
   const int      m,
   const int      n,
   const double * cs_expected,
   const double * weights,
   const double   threshold
)
#else
int HPL_sdc_verify_trailing( A, lda, m, n, cs_expected, weights, threshold )
   const double * A; const int lda, m, n;
   const double * cs_expected; const double * weights; const double threshold;
#endif
{
/*
 * Purpose
 * =======
 * Verify trailing matrix checksums by full recomputation from matrix data.
 * Returns number of mismatched columns (0 if all match).
 */
   int j, failed = 0;
   double cs_comp;

   if( !A || !cs_expected ) return 0;

   for( j = 0; j < n; j++ )
   {
      HPL_sdc_panel_checksum( A + (size_t)j * lda, lda, m, 1, weights, &cs_comp );
      if( HPL_sdc_verify_checksum( cs_expected[j], cs_comp, threshold ) )
         failed++;
   }
   return failed;
}

#endif /* HPL_SDC_CHECK */

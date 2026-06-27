/*
 * HPL_sdc_verify.c - Verification routines for SDC detection
 */
#include "hpl.h"

#ifdef HPL_SDC_CHECK

#ifdef STDC_HEADERS
int HPL_sdc_verify_checksum
(
   double cs_expected,
   double cs_computed,
   double threshold
)
#else
int HPL_sdc_verify_checksum( cs_expected, cs_computed, threshold )
   double cs_expected, cs_computed, threshold;
#endif
{
/*
 * Purpose
 * =======
 * Compare two checksums with relative threshold.
 * Returns HPL_SUCCESS (0) if match, 1 if mismatch (SDC detected).
 */
   double denom, dev;

   dev = cs_computed - cs_expected;
   if( dev < 0.0 ) dev = -dev;

   denom = ( cs_expected >= 0.0 ? cs_expected : -cs_expected );
   if( denom < 1.0 ) denom = 1.0;

   if( dev / denom > threshold )
      return 1;  /* SDC detected */
   else
      return 0;  /* OK */
}

#ifdef STDC_HEADERS
int HPL_sdc_verify_panel
(
   const double * A,
   int            lda,
   int            m,
   int            n,
   const double * weights,
   const double * cs_expected,
   double         threshold
)
#else
int HPL_sdc_verify_panel( A, lda, m, n, weights, cs_expected, threshold )
   const double * A; int lda, m, n;
   const double * weights; const double * cs_expected; double threshold;
#endif
{
/*
 * Purpose
 * =======
 * Verify panel checksums: recompute per-column checksums from matrix
 * data and compare against expected values.
 *
 * Returns HPL_SUCCESS (0) if all columns match, 1 if any mismatch.
 */
   int k;
   double cs_computed;

   for( k = 0; k < n; k++ )
   {
      double s = 0.0;
      int i;
      for( i = 0; i < m; i++ )
         s += weights[i] * A[i + k * lda];
      cs_computed = s;

      if( HPL_sdc_verify_checksum( cs_expected[k], cs_computed, threshold ) )
         return 1;
   }
   return 0;
}

#ifdef STDC_HEADERS
int HPL_sdc_verify_trailing
(
   const double * A,
   int            lda,
   int            m,
   int            n,
   const double * cs_expected,
   double         threshold
)
#else
int HPL_sdc_verify_trailing( A, lda, m, n, cs_expected, threshold )
   const double * A; int lda, m, n;
   const double * cs_expected; double threshold;
#endif
{
/*
 * Purpose
 * =======
 * Verify trailing matrix checksums by full recomputation from matrix
 * data and comparison against expected (incrementally updated) values.
 *
 * Uses uniform weights (sum of all rows) for simplicity and to avoid
 * issues with pivoting permutations.
 *
 * Returns HPL_SUCCESS (0) if match, 1 if mismatch.
 */
   int j;

   for( j = 0; j < n; j++ )
   {
      double s = 0.0;
      int i;
      for( i = 0; i < m; i++ )
         s += A[i + j * lda];
      
      if( HPL_sdc_verify_checksum( cs_expected[j], s, threshold ) )
         return 1;
   }
   return 0;
}

#endif /* HPL_SDC_CHECK */

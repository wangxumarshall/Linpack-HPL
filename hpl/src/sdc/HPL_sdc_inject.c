/*
 * HPL_sdc_inject.c - Fault injection routines for SDC testing
 */
#include "hpl.h"

#if defined(HPL_SDC_CHECK) && defined(HPL_SDC_INJECT)

#include <stdlib.h>
#include <math.h>

#ifdef STDC_HEADERS
void HPL_sdc_inject_bitflip
(
   double * A,
   int      index,
   int      bit_pos
)
#else
void HPL_sdc_inject_bitflip( A, index, bit_pos )
   double * A; int index, bit_pos;
#endif
{
/*
 * Purpose
 * =======
 * Simulate a single bit flip in a double-precision value at A[index].
 */
   unsigned char * bytes = (unsigned char *)&A[index];
   int byte_idx = ( bit_pos / 8 ) % (int)sizeof(double);
   int bit_idx  = bit_pos % 8;
   bytes[byte_idx] ^= (unsigned char)(1 << bit_idx);
}

#ifdef STDC_HEADERS
void HPL_sdc_inject_random
(
   double * A,
   int      n,
   double   rate
)
#else
void HPL_sdc_inject_random( A, n, rate )
   double * A; int n; double rate;
#endif
{
/*
 * Purpose
 * =======
 * Inject random corruption at a given rate (fraction of elements).
 * Each corrupted element is replaced with a random value.
 */
   int i;
   for( i = 0; i < n; i++ )
   {
      if( (double)rand() / (double)RAND_MAX < rate )
      {
         A[i] = ( (double)rand() / (double)RAND_MAX - 0.5 ) * 1.0e10;
      }
   }
}

#ifdef STDC_HEADERS
void HPL_sdc_inject_at
(
   double * A,
   int      index,
   int      mode,
   double   value
)
#else
void HPL_sdc_inject_at( A, index, mode, value )
   double * A; int index, mode; double value;
#endif
{
/*
 * Purpose
 * =======
 * Inject a specific fault at A[index].
 * mode 0: replace with value
 * mode 1: add value (small drift)
 * mode 2: set to zero (stuck-at-zero)
 * mode 3: flip sign
 * mode 4: set to NaN
 * mode 5: set to Inf
 */
   switch( mode )
   {
   case 0: A[index] = value; break;
   case 1: A[index] += value; break;
   case 2: A[index] = 0.0; break;
   case 3: A[index] = -A[index]; break;
   case 4: A[index] = 0.0 / 0.0; break;  /* NaN */
   case 5: A[index] = 1.0 / 0.0; break;  /* Inf */
   default: break;
   }
}

#endif /* HPL_SDC_CHECK && HPL_SDC_INJECT */

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
   const int index,
   const int bit_pos
)
#else
void HPL_sdc_inject_bitflip( A, index, bit_pos )
   double * A; const int index, bit_pos;
#endif
{
/*
 * Purpose
 * =======
 * Simulate a single bit flip in a double-precision value at A[index].
 * Obey strict aliasing via union.
 */
   union {
      double d;
      unsigned char b[sizeof(double)];
   } u;
   unsigned int byte_idx = ( (unsigned int)bit_pos / 8U ) % (unsigned int)sizeof(double);
   unsigned int bit_idx  = (unsigned int)bit_pos % 8U;

   if( !A || index < 0 ) return;

   u.d = A[index];
   u.b[byte_idx] ^= (unsigned char)( 1U << bit_idx );
   A[index] = u.d;
}

#ifdef STDC_HEADERS
void HPL_sdc_inject_random
(
   double * A,
   const int n,
   const double rate
)
#else
void HPL_sdc_inject_random( A, n, rate )
   double * A; const int n; const double rate;
#endif
{
/*
 * Purpose
 * =======
 * Inject random corruption at a given rate (fraction of elements).
 * Ensures at least one element is corrupted if rate > 0 and n > 0.
 */
   int i, corrupted = 0;
   if( !A || n <= 0 || rate <= 0.0 ) return;
   for( i = 0; i < n; i++ )
   {
      if( (double)rand() / (double)RAND_MAX < rate )
      {
         A[i] = ( (double)rand() / (double)RAND_MAX - 0.5 ) * 1.0e10;
         corrupted++;
      }
   }
   if( corrupted == 0 )
   {
      int idx = rand() % n;
      A[idx] = ( (double)rand() / (double)RAND_MAX - 0.5 ) * 1.0e10;
   }
}

#ifdef STDC_HEADERS
void HPL_sdc_inject_at
(
   double * A,
   const int index,
   const int mode,
   const double value
)
#else
void HPL_sdc_inject_at( A, index, mode, value )
   double * A; const int index, mode; const double value;
#endif
{
/*
 * Purpose
 * =======
 * Inject a specific fault at A[index].
 */
   if( !A || index < 0 ) return;
   switch( mode )
   {
   case 0: A[index] = value; break;
   case 1: A[index] += value; break;
   case 2: A[index] = 0.0; break;
   case 3: A[index] = -A[index]; break;
   case 4: A[index] = NAN; break;
   case 5: A[index] = INFINITY; break;
   default: break;
   }
}

#endif /* HPL_SDC_CHECK && HPL_SDC_INJECT */

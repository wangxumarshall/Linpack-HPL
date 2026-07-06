/*
 * HPL_sdc_checksum.c - Core checksum computation routines for SDC detection
 */
#include "hpl.h"

#ifdef HPL_SDC_CHECK

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
   double       * cs_out
)
#else
void HPL_sdc_compute_bcast_checksum( L2, ldl2, ml2, L1, jb_l1, DPIV, jb, cs_out )
   const double * L2; const int ldl2, ml2; const double * L1; const int jb_l1;
   const double * DPIV; const int jb; double * cs_out;
#endif
{
/*
 * Purpose
 * =======
 * Compute unweighted checksum of the broadcast buffer (L2 + L1 + DPIV)
 * using robust Kahan compensated summation to verify communication integrity.
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
            y = L2[i + (size_t)k * ldl2] - c;
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

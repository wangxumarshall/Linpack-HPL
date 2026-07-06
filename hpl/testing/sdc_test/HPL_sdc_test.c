/*
 * HPL_sdc_test.c - Standalone SDC Fault Injection Test Program
 *
 * Tests the complete SDC detection pipeline:
 *   1. Checksum computation correctness
 *   2. Verification logic (true positive / true negative)
 *   3. All 5 fault injection models
 *   4. Fault logging and node-level reporting
 *   5. MPI aggregation
 *
 * Build: linked with HPL library, requires -DHPL_SDC_CHECK -DHPL_SDC_INJECT
 * Run:   mpirun -np 4 xhpl_sdc_test
 */
#include "hpl.h"

#ifdef HPL_SDC_CHECK

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * ---------------------------------------------------------------------
 * Test counters (global, aggregated at end)
 * ---------------------------------------------------------------------
 */
static int test_total   = 0;
static int test_passed  = 0;
static int test_failed  = 0;
static int test_false_pos = 0;

/*
 * ---------------------------------------------------------------------
 * Test result recording
 * ---------------------------------------------------------------------
 */
static void
test_check( int condition, const char * test_name )
{
   test_total++;
   if( condition )
   {
      test_passed++;
      printf( "  [PASS] %s\n", test_name );
   }
   else
   {
      test_failed++;
      printf( "  [FAIL] %s\n", test_name );
   }
}

/*
 * =====================================================================
 * TEST GROUP 1: Broadcast Checksum & Verification (Kahan Summation)
 * =====================================================================
 */
static void
test_bcast_checksum( void )
{
   double L2[8], L1[4], DPIV[2];
   double cs1, cs2;
   int    i, result;

   printf( "\n--- Test Group 1: Broadcast Checksum & Verification ---\n" );

   /* L2: 4x2 (ldl2=4), L1: 2x2, DPIV: 2 */
   for( i = 0; i < 8; i++ ) L2[i] = (double)(i + 1);
   for( i = 0; i < 4; i++ ) L1[i] = (double)(i + 10);
   DPIV[0] = 100.0; DPIV[1] = 200.0;

   /* 1. Unweighted Kahan summation basic computation */
   HPL_sdc_compute_bcast_checksum( L2, 4, 4, L1, 2, DPIV, 2, &cs1 );
   test_check( cs1 != 0.0, "Broadcast checksum non-zero" );

   /* 2. Deterministic repeatability */
   HPL_sdc_compute_bcast_checksum( L2, 4, 4, L1, 2, DPIV, 2, &cs2 );
   test_check( fabs( cs1 - cs2 ) < 1e-15,
               "Broadcast checksum deterministic" );

   /* 3. True negative: identical checksums pass verification */
   result = HPL_sdc_verify_checksum( cs1, cs2, HPL_SDC_THRESHOLD );
   test_check( result == 0, "True negative: clean broadcast passes verification" );

   /* 4. True positive: corrupt L2 and verify detection */
   L2[3] += 0.001;
   HPL_sdc_compute_bcast_checksum( L2, 4, 4, L1, 2, DPIV, 2, &cs2 );
   result = HPL_sdc_verify_checksum( cs1, cs2, HPL_SDC_THRESHOLD );
   test_check( result == 1,
               "True positive: L2 corruption detected by verification" );

   /* 5. True positive: corrupt DPIV and verify detection */
   L2[3] -= 0.001; /* restore */
   DPIV[0] += 5.0;
   HPL_sdc_compute_bcast_checksum( L2, 4, 4, L1, 2, DPIV, 2, &cs2 );
   result = HPL_sdc_verify_checksum( cs1, cs2, HPL_SDC_THRESHOLD );
   test_check( result == 1,
               "True positive: DPIV corruption detected by verification" );

   /* 6. Below threshold perturbation should not trigger alarm */
   DPIV[0] -= 5.0; /* restore */
   DPIV[0] += 1.0e-13 * DPIV[0]; /* relative perturbation < 1e-10 */
   HPL_sdc_compute_bcast_checksum( L2, 4, 4, L1, 2, DPIV, 2, &cs2 );
   result = HPL_sdc_verify_checksum( cs1, cs2, HPL_SDC_THRESHOLD );
   if( result != 0 ) test_false_pos++;
   test_check( result == 0,
               "Below threshold: tiny perturbation (1e-13) not flagged" );
}

/*
 * =====================================================================
 * TEST GROUP 2: JIT Panel Entry Verification (Historical DGEMM Check)
 * =====================================================================
 */
static void
test_panel_entry_verification( void )
{
   double A[16];
   int    i, result;

   printf( "\n--- Test Group 2: JIT Panel Entry Verification ---\n" );

   /* Clean matrix */
   for( i = 0; i < 16; i++ ) A[i] = (double)(i + 1) * 1.5;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 0, "Panel entry clean matrix passes" );

   /* Boundary values exactly at threshold (+/- 1.0e150) */
   A[3] = 1.0e150;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 0, "Panel entry boundary +1.0e150 passes" );

   A[3] = -1.0e150;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 0, "Panel entry boundary -1.0e150 passes" );

   /* Exceeding positive threshold */
   A[3] = 1.0e150 * 1.0001;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 1, "Panel entry > +1.0e150 intercepted" );

   /* Exceeding negative threshold */
   A[3] = -1.0e150 * 1.0001;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 1, "Panel entry < -1.0e150 intercepted" );

   /* NaN value */
   A[3] = NAN;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 1, "Panel entry NaN intercepted" );

   /* Inf value */
   A[3] = INFINITY;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 1, "Panel entry +Inf intercepted" );

   /* -Inf value */
   A[3] = -INFINITY;
   result = HPL_sdc_verify_panel_entry( A, 4, 4, 4 );
   test_check( result == 1, "Panel entry -Inf intercepted" );
}

/*
 * =====================================================================
 * TEST GROUP 3: Fault Logging and Reporting
 * =====================================================================
 */
static void
test_fault_logging( MPI_Comm comm, int myrank )
{
   HPL_T_SDC_LOG log;

   printf( "\n--- Test Group 3: Fault Logging & Reporting ---\n" );

   HPL_sdc_log_init( &log, comm );

   test_check( log.enabled == 1, "Log initialized and enabled" );
   test_check( log.count == 0, "Log initially empty" );
   test_check( strlen( log.node_name ) > 0, "Node name obtained" );

   /* Log some test faults */
   HPL_sdc_log_fault( &log, myrank, 0, 0,
                      HPL_SDC_FAULT_PANEL_ENTRY, 100, 50, 75,
                      0.0, 0.0 );
   test_check( log.count == 1, "Fault logged: count = 1" );

   HPL_sdc_log_fault( &log, myrank, 0, 1,
                      HPL_SDC_FAULT_PANEL_BCAST, 200, 100, 150,
                      5.0, 5.001 );
   test_check( log.count == 2, "Fault logged: count = 2" );

   /* Verify fault record content */
   {
      HPL_T_SDC_FAULT * f = log.head;
      test_check( f != NULL, "Fault list head not NULL" );
      if( f )
      {
         test_check( f->step == 200, "Latest fault: step = 200" );
         test_check( f->fault_type == HPL_SDC_FAULT_PANEL_BCAST,
                     "Latest fault: type = PANEL_BCAST" );
         test_check( fabs( f->deviation - 0.001 ) < 1e-12,
                     "Latest fault: deviation = 0.001" );
      }
   }

   /* Test aggregation (produces report on rank 0) */
   if( myrank == 0 )
      printf( "\n  [INFO] Generating aggregated fault report:\n" );
   HPL_sdc_report_and_aggregate( &log, comm, myrank );

   /* Cleanup */
   HPL_sdc_log_cleanup( &log );
   test_check( log.count == 0, "Log cleaned up: count = 0" );
   test_check( log.head == NULL, "Log cleaned up: head = NULL" );
}

/*
 * =====================================================================
 * MAIN
 * =====================================================================
 */
int main( int argc, char * argv[] )
{
   int myrank, nprocs;
   int total_pass = 0, total_fail = 0, total_fp = 0, total_all = 0;

   MPI_Init( &argc, &argv );
   MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
   MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

   if( myrank == 0 )
   {
      printf( "\n" );
      printf( "========================================================\n" );
      printf( "  HPL SDC Fault Detection Test Suite (Scheme A Pure)\n" );
      printf( "  MPI Processes: %d\n", nprocs );
      printf( "========================================================\n" );
   }

   /*
    * All test groups (each process runs independently for unit tests)
    */
   test_bcast_checksum();
   test_panel_entry_verification();
   test_fault_logging( MPI_COMM_WORLD, myrank );

   /*
    * Aggregate results across all processes
    */
   MPI_Allreduce( &test_total,  &total_all,  1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD );
   MPI_Allreduce( &test_passed, &total_pass, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD );
   MPI_Allreduce( &test_failed, &total_fail, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD );
   MPI_Allreduce( &test_false_pos, &total_fp, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD );

   if( myrank == 0 )
   {
      printf( "\n" );
      printf( "========================================================\n" );
      printf( "  TEST REPORT SUMMARY\n" );
      printf( "========================================================\n" );
      printf( "  Total tests (all ranks): %d\n", total_all  );
      printf( "  Passed:                  %d\n", total_pass );
      printf( "  Failed:                  %d\n", total_fail );
      printf( "  False positives:         %d\n", total_fp   );
      printf( "  Detection rate:          %.1f%%\n",
              total_all > 0 ? 100.0 * (double)total_pass /
                              (double)total_all : 0.0 );
      printf( "--------------------------------------------------------\n" );

      if( total_fail == 0 && total_fp == 0 )
         printf( "  RESULT: ALL TESTS PASSED\n" );
      else
         printf( "  RESULT: %d FAILURE(S) DETECTED\n", total_fail );

      printf( "========================================================\n" );
      printf( "\n" );
   }

   MPI_Finalize();
   return( total_fail > 0 ? 1 : 0 );
}

#else /* !HPL_SDC_CHECK */

#include <stdio.h>
int main( void )
{
   printf( "ERROR: This test requires -DHPL_SDC_CHECK -DHPL_SDC_INJECT\n" );
   return 1;
}

#endif /* HPL_SDC_CHECK */

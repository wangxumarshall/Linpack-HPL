/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    SDC (Silent Data Corruption) Detection Module
 *    Added as part of SDC detection enhancement                      
 */ 
#ifndef HPL_SDC_H
#define HPL_SDC_H
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_pmisc.h"
#include <string.h>
#include <math.h>

#ifdef HPL_SDC_CHECK
/*
 * ---------------------------------------------------------------------
 * SDC configuration defaults (can be overridden at compile time)
 * ---------------------------------------------------------------------
 */
#ifndef HPL_SDC_BCAST_VERIFY
#define HPL_SDC_BCAST_VERIFY          1
#endif

#ifndef HPL_SDC_THRESHOLD
#define HPL_SDC_THRESHOLD             1.0e-10
#endif

#ifndef HPL_SDC_WEIGHT_WINDOW
#define HPL_SDC_WEIGHT_WINDOW         16
#endif

#ifndef HPL_SDC_NODE_NAME_LEN
#define HPL_SDC_NODE_NAME_LEN         64
#endif

/*
 * ---------------------------------------------------------------------
 * SDC fault type enumeration
 * ---------------------------------------------------------------------
 */
typedef enum
{
   HPL_SDC_FAULT_PANEL_BCAST    = 0,   /* panel broadcast corruption  */
   HPL_SDC_FAULT_PANEL_FACT     = 1,   /* panel factorization corrupt  */
   HPL_SDC_FAULT_PANEL_ENTRY    = 2,   /* panel entry check (hist DGEMM) */
   HPL_SDC_FAULT_BACK_SOLVE     = 3,   /* back substitution corruption */
   HPL_SDC_FAULT_BROADCAST      = 4,   /* comm layer broadcast corrupt */
   HPL_SDC_FAULT_UNKNOWN        = 5,   /* unknown type                 */
   HPL_SDC_FAULT_COUNT          = 6    /* count of fault types         */
} HPL_T_SDC_FAULT_TYPE;

/*
 * ---------------------------------------------------------------------
 * SDC fault record (linked list node)
 * ---------------------------------------------------------------------
 */
typedef struct HPL_S_SDC_FAULT
{
   int                  mpi_rank;
   int                  grid_row;
   int                  grid_col;
   char                 node_name[HPL_SDC_NODE_NAME_LEN];
   HPL_T_SDC_FAULT_TYPE fault_type;
   int                  step;
   int                  global_row;
   int                  global_col;
   double               cs_expected;
   double               cs_computed;
   double               deviation;
   struct HPL_S_SDC_FAULT * next;
} HPL_T_SDC_FAULT;

/*
 * ---------------------------------------------------------------------
 * SDC fault log (per-process)
 * ---------------------------------------------------------------------
 */
typedef struct HPL_S_SDC_LOG
{
   HPL_T_SDC_FAULT * head;
   int               count;
   int               enabled;
   char              node_name[HPL_SDC_NODE_NAME_LEN];
} HPL_T_SDC_LOG;

extern HPL_T_SDC_LOG HPL_sdc_global_log;
#define sdc_log_global HPL_sdc_global_log

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */

/* Checksum computation (HPL_sdc_checksum.c) */
void   HPL_sdc_init_weights
STDC_ARGS( ( double *, const int ) );
double HPL_sdc_col_checksum
STDC_ARGS( ( const double *, const int, const int, const int, const double * ) );
void   HPL_sdc_panel_checksum
STDC_ARGS( ( const double *, const int, const int, const int, const double *, double * ) );
void   HPL_sdc_compute_bcast_checksum
STDC_ARGS( ( const double *, const int, const int, const double *, const int, const double *,
              const int, const double *, double * ) );

/* Verification (HPL_sdc_verify.c) */
int    HPL_sdc_verify_checksum
STDC_ARGS( ( const double, const double, const double ) );
int    HPL_sdc_verify_panel
STDC_ARGS( ( const double *, const int, const int, const int, const double *,
              const double *, const double ) );
int    HPL_sdc_verify_panel_entry
STDC_ARGS( ( const double *, const int, const int, const int ) );

/* Fault logging and reporting (HPL_sdc_report.c) */
void   HPL_sdc_log_init
STDC_ARGS( ( HPL_T_SDC_LOG *, MPI_Comm ) );
void   HPL_sdc_log_fault
STDC_ARGS( ( HPL_T_SDC_LOG *, const int, const int, const int,
              const HPL_T_SDC_FAULT_TYPE, const int, const int, const int,
              const double, const double ) );
void   HPL_sdc_report_and_aggregate
STDC_ARGS( ( HPL_T_SDC_LOG *, MPI_Comm, const int ) );
void   HPL_sdc_log_cleanup
STDC_ARGS( ( HPL_T_SDC_LOG * ) );

/* Fault injection - testing only (HPL_sdc_inject.c) */
#ifdef HPL_SDC_INJECT
void   HPL_sdc_inject_bitflip
STDC_ARGS( ( double *, const int, const int ) );
void   HPL_sdc_inject_random
STDC_ARGS( ( double *, const int, const double ) );
void   HPL_sdc_inject_at
STDC_ARGS( ( double *, const int, const int, const double ) );
#endif

#endif /* HPL_SDC_CHECK */

#endif
/*
 * End of hpl_sdc.h
 */

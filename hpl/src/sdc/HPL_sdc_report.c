/*
 * HPL_sdc_report.c - Node-level SDC fault logging and aggregation
 *
 * Supports reporting SDC faults with specific core/node identification
 * for quick replacement of faulty nodes.
 */
#include "hpl.h"

#ifdef HPL_SDC_CHECK

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global SDC fault log accessible across all HPL routines */
HPL_T_SDC_LOG HPL_sdc_global_log;

#ifdef STDC_HEADERS
void HPL_sdc_log_init
(
   HPL_T_SDC_LOG * log,
   MPI_Comm        comm
)
#else
void HPL_sdc_log_init( log, comm )
   HPL_T_SDC_LOG * log; MPI_Comm comm;
#endif
{
/*
 * Purpose
 * =======
 * Initialize the SDC fault log for this process.
 * Obtains the physical node name via MPI_Get_processor_name.
 */
   char name[MPI_MAX_PROCESSOR_NAME];
   int  namelen;

   MPI_Get_processor_name( name, &namelen );
   if( namelen >= HPL_SDC_NODE_NAME_LEN ) namelen = HPL_SDC_NODE_NAME_LEN - 1;
   strncpy( log->node_name, name, (size_t)namelen );
   log->node_name[namelen] = '\0';

   log->head    = NULL;
   log->count   = 0;
   log->enabled = 1;
}

#ifdef STDC_HEADERS
void HPL_sdc_log_fault
(
   HPL_T_SDC_LOG      * log,
   int                  mpi_rank,
   int                  grid_row,
   int                  grid_col,
   HPL_T_SDC_FAULT_TYPE type,
   int                  step,
   int                  global_row,
   int                  global_col,
   double               cs_expected,
   double               cs_computed
)
#else
void HPL_sdc_log_fault( log, mpi_rank, grid_row, grid_col,
                         type, step, global_row, global_col,
                         cs_expected, cs_computed )
   HPL_T_SDC_LOG * log; int mpi_rank, grid_row, grid_col;
   HPL_T_SDC_FAULT_TYPE type; int step, global_row, global_col;
   double cs_expected, cs_computed;
#endif
{
/*
 * Purpose
 * =======
 * Record a single SDC fault into the local fault log (linked list).
 * O(1) insertion.
 */
   HPL_T_SDC_FAULT * node;
   double dev;

   if( !log->enabled ) return;

   node = (HPL_T_SDC_FAULT *)malloc( sizeof(HPL_T_SDC_FAULT) );
   if( node == NULL ) return;  /* silently skip if OOM */

   dev = cs_computed - cs_expected;
   if( dev < 0.0 ) dev = -dev;

   node->mpi_rank   = mpi_rank;
   node->grid_row   = grid_row;
   node->grid_col   = grid_col;
   strncpy( node->node_name, log->node_name, HPL_SDC_NODE_NAME_LEN - 1 );
   node->node_name[HPL_SDC_NODE_NAME_LEN - 1] = '\0';
   node->fault_type = type;
   node->step       = step;
   node->global_row = global_row;
   node->global_col = global_col;
   node->cs_expected = cs_expected;
   node->cs_computed = cs_computed;
   node->deviation  = dev;

   /* Insert at head */
   node->next = log->head;
   log->head  = node;
   log->count++;
}

static const char * HPL_sdc_fault_type_str
(
   HPL_T_SDC_FAULT_TYPE type
)
{
   switch( type )
   {
   case HPL_SDC_FAULT_PANEL_BCAST:  return "PANEL_BCAST";
   case HPL_SDC_FAULT_PANEL_FACT:   return "PANEL_FACT";
   case HPL_SDC_FAULT_TRAIL_UPDATE: return "TRAIL_UPDATE";
   case HPL_SDC_FAULT_BACK_SOLVE:   return "BACK_SOLVE";
   case HPL_SDC_FAULT_BROADCAST:    return "BROADCAST";
   default:                         return "UNKNOWN";
   }
}

/*
 * Serialization is now done per-field (see HPL_sdc_report_and_aggregate),
 * so no packed structure is needed. Each field is gathered independently
 * with its proper MPI type, avoiding any alignment/byte-count ambiguity.
 */

#ifdef STDC_HEADERS
void HPL_sdc_report_and_aggregate
(
   HPL_T_SDC_LOG * local_log,
   MPI_Comm        comm,
   int             my_rank
)
#else
void HPL_sdc_report_and_aggregate( local_log, comm, my_rank )
   HPL_T_SDC_LOG * local_log; MPI_Comm comm; int my_rank;
#endif
{
/*
 * Purpose
 * =======
 * Aggregate fault logs from all processes to rank 0 and output
 * a comprehensive report with per-node and per-type summaries.
 *
 * Communication: one MPI_Allreduce (int) for total count, then
 * per-field MPI_Gatherv. We gather each field independently with
 * its proper MPI type to avoid any packed-structure alignment /
 * byte-count ambiguity.
 */
   int total_faults = 0, local_count = local_log->count;
   int nprocs;
   int * counts = NULL, * displs = NULL;
   HPL_T_SDC_FAULT * p;
   int i, idx;

   /* Per-field local buffers */
   int    * l_mpi_rank   = NULL;
   int    * l_grid_row   = NULL;
   int    * l_grid_col   = NULL;
   int    * l_fault_type = NULL;
   int    * l_step       = NULL;
   int    * l_global_row = NULL;
   int    * l_global_col = NULL;
   double * l_cs_expected= NULL;
   double * l_cs_computed= NULL;
   double * l_deviation  = NULL;
   char   * l_node_name  = NULL;  /* local_count * NODE_NAME_LEN */

   /* Per-field global buffers (rank 0 only) */
   int    * g_mpi_rank   = NULL;
   int    * g_grid_row   = NULL;
   int    * g_grid_col   = NULL;
   int    * g_fault_type = NULL;
   int    * g_step       = NULL;
   int    * g_global_row = NULL;
   int    * g_global_col = NULL;
   double * g_cs_expected= NULL;
   double * g_cs_computed= NULL;
   double * g_deviation  = NULL;
   char   * g_node_name  = NULL;

   MPI_Comm_size( comm, &nprocs );

   /* Step 1: Global total */
   MPI_Allreduce( &local_count, &total_faults, 1, MPI_INT, MPI_SUM, comm );

   if( total_faults == 0 ) return;  /* No faults, nothing to report */

   /* Step 2: Pack local faults into per-field arrays */
   if( local_count > 0 )
   {
      l_mpi_rank    = (int *)   malloc( local_count * sizeof(int) );
      l_grid_row    = (int *)   malloc( local_count * sizeof(int) );
      l_grid_col    = (int *)   malloc( local_count * sizeof(int) );
      l_fault_type  = (int *)   malloc( local_count * sizeof(int) );
      l_step        = (int *)   malloc( local_count * sizeof(int) );
      l_global_row  = (int *)   malloc( local_count * sizeof(int) );
      l_global_col  = (int *)   malloc( local_count * sizeof(int) );
      l_cs_expected = (double *)malloc( local_count * sizeof(double) );
      l_cs_computed = (double *)malloc( local_count * sizeof(double) );
      l_deviation   = (double *)malloc( local_count * sizeof(double) );
      l_node_name   = (char *)  malloc( (size_t)local_count * HPL_SDC_NODE_NAME_LEN );

      if( l_mpi_rank && l_grid_row && l_grid_col && l_fault_type &&
          l_step && l_global_row && l_global_col && l_cs_expected &&
          l_cs_computed && l_deviation && l_node_name )
      {
         p = local_log->head; idx = 0;
         while( p && idx < local_count )
         {
            l_mpi_rank[idx]    = p->mpi_rank;
            l_grid_row[idx]    = p->grid_row;
            l_grid_col[idx]    = p->grid_col;
            l_fault_type[idx]  = (int)p->fault_type;
            l_step[idx]        = p->step;
            l_global_row[idx]  = p->global_row;
            l_global_col[idx]  = p->global_col;
            l_cs_expected[idx] = p->cs_expected;
            l_cs_computed[idx] = p->cs_computed;
            l_deviation[idx]   = p->deviation;
            strncpy( l_node_name + idx * HPL_SDC_NODE_NAME_LEN,
                     p->node_name, HPL_SDC_NODE_NAME_LEN - 1 );
            l_node_name[ idx * HPL_SDC_NODE_NAME_LEN + HPL_SDC_NODE_NAME_LEN - 1 ] = '\0';
            p = p->next; idx++;
         }
      }
   }

   /* Step 3: Gather counts and displacements (in fault records) */
   counts = (int *)malloc( (size_t)nprocs * sizeof(int) );
   displs = (int *)malloc( (size_t)nprocs * sizeof(int) );

   MPI_Allgather( &local_count, 1, MPI_INT, counts, 1, MPI_INT, comm );

   displs[0] = 0;
   for( i = 1; i < nprocs; i++ )
      displs[i] = displs[i-1] + counts[i-1];

   /* Step 4: Allocate global per-field buffers on rank 0 */
   if( my_rank == 0 && total_faults > 0 )
   {
      g_mpi_rank    = (int *)   malloc( total_faults * sizeof(int) );
      g_grid_row    = (int *)   malloc( total_faults * sizeof(int) );
      g_grid_col    = (int *)   malloc( total_faults * sizeof(int) );
      g_fault_type  = (int *)   malloc( total_faults * sizeof(int) );
      g_step        = (int *)   malloc( total_faults * sizeof(int) );
      g_global_row  = (int *)   malloc( total_faults * sizeof(int) );
      g_global_col  = (int *)   malloc( total_faults * sizeof(int) );
      g_cs_expected = (double *)malloc( total_faults * sizeof(double) );
      g_cs_computed = (double *)malloc( total_faults * sizeof(double) );
      g_deviation   = (double *)malloc( total_faults * sizeof(double) );
      g_node_name   = (char *)  malloc( (size_t)total_faults * HPL_SDC_NODE_NAME_LEN );
   }

   /* Step 5: Gather each field independently with correct MPI type */
   MPI_Gatherv( l_mpi_rank,    local_count, MPI_INT,    g_mpi_rank,    counts, displs, MPI_INT,    0, comm );
   MPI_Gatherv( l_grid_row,    local_count, MPI_INT,    g_grid_row,    counts, displs, MPI_INT,    0, comm );
   MPI_Gatherv( l_grid_col,    local_count, MPI_INT,    g_grid_col,    counts, displs, MPI_INT,    0, comm );
   MPI_Gatherv( l_fault_type,  local_count, MPI_INT,    g_fault_type,  counts, displs, MPI_INT,    0, comm );
   MPI_Gatherv( l_step,        local_count, MPI_INT,    g_step,        counts, displs, MPI_INT,    0, comm );
   MPI_Gatherv( l_global_row,  local_count, MPI_INT,    g_global_row,  counts, displs, MPI_INT,    0, comm );
   MPI_Gatherv( l_global_col,  local_count, MPI_INT,    g_global_col,  counts, displs, MPI_INT,    0, comm );
   MPI_Gatherv( l_cs_expected, local_count, MPI_DOUBLE, g_cs_expected, counts, displs, MPI_DOUBLE, 0, comm );
   MPI_Gatherv( l_cs_computed, local_count, MPI_DOUBLE, g_cs_computed, counts, displs, MPI_DOUBLE, 0, comm );
   MPI_Gatherv( l_deviation,   local_count, MPI_DOUBLE, g_deviation,   counts, displs, MPI_DOUBLE, 0, comm );
   /* node_name: counts/displs in records, but element is NODE_NAME_LEN bytes */
   {
      int * nm_counts = (int *)malloc( (size_t)nprocs * sizeof(int) );
      int * nm_displs = (int *)malloc( (size_t)nprocs * sizeof(int) );
      for( i = 0; i < nprocs; i++ ) nm_counts[i] = counts[i] * HPL_SDC_NODE_NAME_LEN;
      nm_displs[0] = 0;
      for( i = 1; i < nprocs; i++ ) nm_displs[i] = nm_displs[i-1] + nm_counts[i-1];
      MPI_Gatherv( l_node_name, local_count * HPL_SDC_NODE_NAME_LEN, MPI_CHAR,
                   g_node_name, nm_counts, nm_displs, MPI_CHAR, 0, comm );
      free( nm_counts ); free( nm_displs );
   }

   /* Step 6: Rank 0 outputs the report */
   if( my_rank == 0 && g_mpi_rank )
   {
      /* Count unique nodes dynamically */
      int num_nodes = 0;
      int max_nodes = total_faults;
      char * node_list   = (char *)malloc( (size_t)max_nodes * HPL_SDC_NODE_NAME_LEN );
      int  * node_faults = (int *) calloc( (size_t)max_nodes, sizeof(int) );
      int  type_counts[6] = {0,0,0,0,0,0};

      HPL_fprintf( stdout, "\n" );
      HPL_fprintf( stdout, "===== SDC FAULT REPORT =====\n" );
      HPL_fprintf( stdout, "Total faults detected: %d\n\n", total_faults );

      /* Print each fault */
      for( i = 0; i < total_faults; i++ )
      {
         HPL_fprintf( stdout, "--- Fault #%d ---\n", i+1 );
         HPL_fprintf( stdout, "  Type:        %s\n",
            HPL_sdc_fault_type_str( (HPL_T_SDC_FAULT_TYPE)g_fault_type[i] ) );
         HPL_fprintf( stdout, "  Step:        %d\n", g_step[i] );
         HPL_fprintf( stdout, "  MPI Rank:    %d\n", g_mpi_rank[i] );
         HPL_fprintf( stdout, "  Grid Pos:    (row=%d, col=%d)\n",
            g_grid_row[i], g_grid_col[i] );
         HPL_fprintf( stdout, "  Node Name:   %s\n",
            g_node_name + i * HPL_SDC_NODE_NAME_LEN );
         HPL_fprintf( stdout, "  Location:    global A[%d, %d]\n",
            g_global_row[i], g_global_col[i] );
         HPL_fprintf( stdout, "  Deviation:   %.3e\n", g_deviation[i] );
         HPL_fprintf( stdout, "  Severity:    %s\n\n",
            ( g_deviation[i] > 1.0e-6 ? "CRITICAL" :
              g_deviation[i] > 1.0e-10 ? "HIGH" : "LOW" ) );

         /* Tally by type */
         if( g_fault_type[i] >= 0 && g_fault_type[i] <= 5 )
            type_counts[g_fault_type[i]]++;

         /* Tally by node */
         if( node_list && node_faults )
         {
            int found = 0, k;
            char * nm = g_node_name + i * HPL_SDC_NODE_NAME_LEN;
            for( k = 0; k < num_nodes; k++ )
            {
               if( strcmp( node_list + k * HPL_SDC_NODE_NAME_LEN, nm ) == 0 )
               { node_faults[k]++; found = 1; break; }
            }
            if( !found && num_nodes < max_nodes )
            {
               strncpy( node_list + num_nodes * HPL_SDC_NODE_NAME_LEN, nm, HPL_SDC_NODE_NAME_LEN - 1 );
               node_list[num_nodes * HPL_SDC_NODE_NAME_LEN + HPL_SDC_NODE_NAME_LEN - 1] = '\0';
               node_faults[num_nodes] = 1;
               num_nodes++;
            }
         }
      }

      /* Summary by node */
      HPL_fprintf( stdout, "--- Summary by Node ---\n" );
      if( node_list && node_faults )
      {
         for( i = 0; i < num_nodes; i++ )
         {
            HPL_fprintf( stdout, "  %s:  %d faults\n",
               node_list + i * HPL_SDC_NODE_NAME_LEN, node_faults[i] );
         }
      }

      /* Summary by type */
      HPL_fprintf( stdout, "\n--- Summary by Fault Type ---\n" );
      HPL_fprintf( stdout, "  TRAIL_UPDATE: %d, PANEL_BCAST: %d, "
         "PANEL_FACT: %d\n",
         type_counts[HPL_SDC_FAULT_TRAIL_UPDATE],
         type_counts[HPL_SDC_FAULT_PANEL_BCAST],
         type_counts[HPL_SDC_FAULT_PANEL_FACT] );
      HPL_fprintf( stdout, "  BACK_SOLVE: %d, BROADCAST: %d, "
         "UNKNOWN: %d\n",
         type_counts[HPL_SDC_FAULT_BACK_SOLVE],
         type_counts[HPL_SDC_FAULT_BROADCAST],
         type_counts[HPL_SDC_FAULT_UNKNOWN] );

      /* Recommendation */
      HPL_fprintf( stdout, "\nRECOMMENDATION: Replace nodes with >10 faults:\n  " );
      if( node_list && node_faults )
      {
         int printed = 0, k;
         for( k = 0; k < num_nodes; k++ )
         {
            if( node_faults[k] > 10 )
            {
               if( printed ) HPL_fprintf( stdout, ", " );
               HPL_fprintf( stdout, "%s", node_list + k * HPL_SDC_NODE_NAME_LEN );
               printed = 1;
            }
         }
         if( !printed ) HPL_fprintf( stdout, "(none)" );
      }
      else
      {
         HPL_fprintf( stdout, "(none)" );
      }
      HPL_fprintf( stdout, "\n==============================\n\n" );

      if( node_list )   free( node_list );
      if( node_faults ) free( node_faults );
   }

   /* Cleanup */
   if( l_mpi_rank )    free( l_mpi_rank );
   if( l_grid_row )    free( l_grid_row );
   if( l_grid_col )    free( l_grid_col );
   if( l_fault_type )  free( l_fault_type );
   if( l_step )        free( l_step );
   if( l_global_row )  free( l_global_row );
   if( l_global_col )  free( l_global_col );
   if( l_cs_expected ) free( l_cs_expected );
   if( l_cs_computed ) free( l_cs_computed );
   if( l_deviation )   free( l_deviation );
   if( l_node_name )   free( l_node_name );
   if( g_mpi_rank )    free( g_mpi_rank );
   if( g_grid_row )    free( g_grid_row );
   if( g_grid_col )    free( g_grid_col );
   if( g_fault_type )  free( g_fault_type );
   if( g_step )        free( g_step );
   if( g_global_row )  free( g_global_row );
   if( g_global_col )  free( g_global_col );
   if( g_cs_expected ) free( g_cs_expected );
   if( g_cs_computed ) free( g_cs_computed );
   if( g_deviation )   free( g_deviation );
   if( g_node_name )   free( g_node_name );
   if( counts )        free( counts );
   if( displs )        free( displs );
}

#ifdef STDC_HEADERS
void HPL_sdc_log_cleanup
(
   HPL_T_SDC_LOG * log
)
#else
void HPL_sdc_log_cleanup( log )
   HPL_T_SDC_LOG * log;
#endif
{
/*
 * Purpose
 * =======
 * Free all fault records in the log.
 */
   HPL_T_SDC_FAULT * p = log->head;
   while( p )
   {
      HPL_T_SDC_FAULT * next = p->next;
      free( p );
      p = next;
   }
   log->head  = NULL;
   log->count = 0;
}

#endif /* HPL_SDC_CHECK */

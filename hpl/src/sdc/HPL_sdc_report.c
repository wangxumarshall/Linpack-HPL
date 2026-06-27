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
 * Serialization buffer for a single fault record (for MPI transport)
 */
typedef struct {
   int    mpi_rank;
   int    grid_row;
   int    grid_col;
   int    fault_type;
   int    step;
   int    global_row;
   int    global_col;
   double cs_expected;
   double cs_computed;
   double deviation;
   char   node_name[HPL_SDC_NODE_NAME_LEN];
} HPL_T_SDC_FAULT_PACKED;

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
 * MPI_Gatherv for the packed fault records.
 */
   int total_faults = 0, local_count = local_log->count;
   int nprocs;
   int * counts = NULL, * displs = NULL;
   HPL_T_SDC_FAULT_PACKED * local_buf = NULL, * all_buf = NULL;
   HPL_T_SDC_FAULT * p;
   int i, idx;

   MPI_Comm_size( comm, &nprocs );

   /* Step 1: Global total */
   MPI_Allreduce( &local_count, &total_faults, 1, MPI_INT, MPI_SUM, comm );

   if( total_faults == 0 ) return;  /* No faults, nothing to report */

   /* Step 2: Pack local faults */
   if( local_count > 0 )
   {
      local_buf = (HPL_T_SDC_FAULT_PACKED *)malloc(
         (size_t)local_count * sizeof(HPL_T_SDC_FAULT_PACKED) );
      if( local_buf )
      {
         p = local_log->head; idx = 0;
         while( p && idx < local_count )
         {
            local_buf[idx].mpi_rank   = p->mpi_rank;
            local_buf[idx].grid_row   = p->grid_row;
            local_buf[idx].grid_col   = p->grid_col;
            local_buf[idx].fault_type = (int)p->fault_type;
            local_buf[idx].step       = p->step;
            local_buf[idx].global_row = p->global_row;
            local_buf[idx].global_col = p->global_col;
            local_buf[idx].cs_expected = p->cs_expected;
            local_buf[idx].cs_computed = p->cs_computed;
            local_buf[idx].deviation  = p->deviation;
            strncpy( local_buf[idx].node_name, p->node_name,
                     HPL_SDC_NODE_NAME_LEN - 1 );
            local_buf[idx].node_name[HPL_SDC_NODE_NAME_LEN-1] = '\0';
            p = p->next; idx++;
         }
      }
   }

   /* Step 3: Gather counts and displacements */
   counts = (int *)malloc( (size_t)nprocs * sizeof(int) );
   displs = (int *)malloc( (size_t)nprocs * sizeof(int) );

   MPI_Allgather( &local_count, 1, MPI_INT, counts, 1, MPI_INT, comm );

   displs[0] = 0;
   for( i = 1; i < nprocs; i++ )
      displs[i] = displs[i-1] + counts[i-1];

   /* Step 4: Gather all packed fault records on rank 0 */
   if( total_faults > 0 )
   {
      all_buf = (HPL_T_SDC_FAULT_PACKED *)malloc(
         (size_t)total_faults * sizeof(HPL_T_SDC_FAULT_PACKED) );
   }

   MPI_Gatherv( local_buf, local_count, MPI_BYTE,
                all_buf, counts, displs, MPI_BYTE,
                0, comm );

   /* Step 5: Rank 0 outputs the report */
   if( my_rank == 0 && all_buf )
   {
      /* Count unique nodes */
      int num_nodes = 0;
      char node_list[256][HPL_SDC_NODE_NAME_LEN];
      int  node_faults[256];
      int  type_counts[6] = {0,0,0,0,0,0};

      memset( node_list, 0, sizeof(node_list) );
      memset( node_faults, 0, sizeof(node_faults) );

      HPL_fprintf( stdout, "\n" );
      HPL_fprintf( stdout, "===== SDC FAULT REPORT =====\n" );
      HPL_fprintf( stdout, "Total faults detected: %d\n\n", total_faults );

      /* Print each fault */
      for( i = 0; i < total_faults; i++ )
      {
         HPL_fprintf( stdout, "--- Fault #%d ---\n", i+1 );
         HPL_fprintf( stdout, "  Type:        %s\n",
            HPL_sdc_fault_type_str( (HPL_T_SDC_FAULT_TYPE)all_buf[i].fault_type ) );
         HPL_fprintf( stdout, "  Step:        %d\n", all_buf[i].step );
         HPL_fprintf( stdout, "  MPI Rank:    %d\n", all_buf[i].mpi_rank );
         HPL_fprintf( stdout, "  Grid Pos:    (row=%d, col=%d)\n",
            all_buf[i].grid_row, all_buf[i].grid_col );
         HPL_fprintf( stdout, "  Node Name:   %s\n", all_buf[i].node_name );
         HPL_fprintf( stdout, "  Location:    global A[%d, %d]\n",
            all_buf[i].global_row, all_buf[i].global_col );
         HPL_fprintf( stdout, "  Deviation:   %.3e\n", all_buf[i].deviation );
         HPL_fprintf( stdout, "  Severity:    %s\n\n",
            ( all_buf[i].deviation > 1.0e-6 ? "CRITICAL" :
              all_buf[i].deviation > 1.0e-10 ? "HIGH" : "LOW" ) );

         /* Tally by type */
         if( all_buf[i].fault_type >= 0 && all_buf[i].fault_type <= 5 )
            type_counts[all_buf[i].fault_type]++;

         /* Tally by node */
         {
            int found = 0, k;
            for( k = 0; k < num_nodes; k++ )
            {
               if( strcmp( node_list[k], all_buf[i].node_name ) == 0 )
               { node_faults[k]++; found = 1; break; }
            }
            if( !found && num_nodes < 256 )
            {
               strncpy( node_list[num_nodes], all_buf[i].node_name,
                        HPL_SDC_NODE_NAME_LEN - 1 );
               node_list[num_nodes][HPL_SDC_NODE_NAME_LEN-1] = '\0';
               node_faults[num_nodes] = 1;
               num_nodes++;
            }
         }
      }

      /* Summary by node */
      HPL_fprintf( stdout, "--- Summary by Node ---\n" );
      for( i = 0; i < num_nodes; i++ )
      {
         HPL_fprintf( stdout, "  %s:  %d faults\n",
            node_list[i], node_faults[i] );
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
      {
         int printed = 0, k;
         for( k = 0; k < num_nodes; k++ )
         {
            if( node_faults[k] > 10 )
            {
               if( printed ) HPL_fprintf( stdout, ", " );
               HPL_fprintf( stdout, "%s", node_list[k] );
               printed = 1;
            }
         }
         if( !printed ) HPL_fprintf( stdout, "(none)" );
      }
      HPL_fprintf( stdout, "\n==============================\n\n" );
   }

   /* Cleanup */
   if( local_buf ) free( local_buf );
   if( all_buf   ) free( all_buf   );
   if( counts    ) free( counts    );
   if( displs    ) free( displs    );
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

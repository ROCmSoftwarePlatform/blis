#include "blis.h"
#ifdef BLIS_ENABLE_AMD_OFFLOAD
#include "bli_offloader.h"
#include <dlfcn.h>
#include <limits.h>
#include "mjson.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>

// The global rntm_t structure. (The definition resides in bli_rntm.c.)
extern rntm_t global_rntm;

// A mutex to allow synchronous access to global_rntm. (The definition
// resides in bli_rntm.c.)
extern bli_pthread_mutex_t global_rntm_mutex;

// PM1 parametrization - in static fields to be able to parse through mjson
static const int BLI_OFF_CPU_A = 0;
static const int BLI_OFF_CPU_B = 1;
static const int BLI_OFF_ACC_A = 2;
static const int BLI_OFF_ACC_B = 3;
static bool   bli_off_mem_trans;
static double bli_off_a_mem;
static double bli_off_b_mem;
static int    bli_off_sgemm_count;
static double bli_off_sgemm[4];
static int    bli_off_sgemm_sq_count;
static double bli_off_sgemm_sq[4];
static int    bli_off_dgemm_count;
static double bli_off_dgemm[4];
static int    bli_off_dgemm_sq_count;
static double bli_off_dgemm_sq[4];
static int    bli_off_cgemm_count;
static double bli_off_cgemm[4];
static int    bli_off_cgemm_sq_count;
static double bli_off_cgemm_sq[4];
static int    bli_off_zgemm_count;
static double bli_off_zgemm[4];
static int    bli_off_zgemm_sq_count;
static double bli_off_zgemm_sq[4];

static const struct json_attr_t json_attrs_bli_off[] = {
    {"mem_transfer_needed",   t_boolean, .addr.boolean = &bli_off_mem_trans,},
    {"a_mem_byte",   t_real, .addr.real = &bli_off_a_mem,},
    {"b_mem_byte",   t_real, .addr.real = &bli_off_b_mem,},
    {"sgemm",        t_array,	.addr.array.element_type = t_real,
				.addr.array.arr.reals = bli_off_sgemm,
				.addr.array.maxlen = 4,
				.addr.array.count = &bli_off_sgemm_count},
    {"sgemm_sq",     t_array,   .addr.array.element_type = t_real,
                                .addr.array.arr.reals = bli_off_sgemm_sq,
                                .addr.array.maxlen = 4,
                                .addr.array.count = &bli_off_sgemm_sq_count},
    {"dgemm",        t_array,   .addr.array.element_type = t_real,
                                .addr.array.arr.reals = bli_off_dgemm,
                                .addr.array.maxlen = 4,
                                .addr.array.count = &bli_off_dgemm_count},
    {"dgemm_sq",     t_array,   .addr.array.element_type = t_real,
                                .addr.array.arr.reals = bli_off_dgemm_sq,
                                .addr.array.maxlen = 4,
                                .addr.array.count = &bli_off_dgemm_sq_count},
    {"cgemm",        t_array,   .addr.array.element_type = t_real,
                                .addr.array.arr.reals = bli_off_cgemm,
                                .addr.array.maxlen = 4,
                                .addr.array.count = &bli_off_cgemm_count},
    {"cgemm_sq",     t_array,   .addr.array.element_type = t_real,
                                .addr.array.arr.reals = bli_off_cgemm_sq,
                                .addr.array.maxlen = 4,
                                .addr.array.count = &bli_off_cgemm_sq_count},
    {"zgemm",        t_array,   .addr.array.element_type = t_real,
                                .addr.array.arr.reals = bli_off_zgemm,
                                .addr.array.maxlen = 4,
                                .addr.array.count = &bli_off_zgemm_count},
    {"zgemm_sq",     t_array,   .addr.array.element_type = t_real,
                                .addr.array.arr.reals = bli_off_zgemm_sq,
                                .addr.array.maxlen = 4,
                                .addr.array.count = &bli_off_zgemm_sq_count},
    {NULL},
};

void bli_offloader_init ( void )
{
	bli_offloader_init_rntm_from_env ( &global_rntm );
}

void bli_offloader_init_rntm_from_env ( rntm_t* rntm )
{
	// allocate struct
	rntm->offloader_state = malloc ( sizeof ( bli_offload_t ) );
	bli_offload_t* config = rntm->offloader_state;
	config->rocblas = NULL;

	char* s_eng = getenv ( "BLIS_OFFLOAD" );
	s_eng = ( s_eng == NULL ) ? "never" : s_eng;
	if ( strcmp ( s_eng, "never" ) == 0 )
	{
		fprintf ( stdout, "Never attempting to offload.\n" );
		config->model = never;
		config->never_offload_dgemm = true;
		config->never_offload_sgemm = true;
		config->never_offload_zgemm = true;
                config->never_offload_cgemm = true;
		config->offload_sgemm_thresh = LLONG_MAX;
		config->offload_dgemm_thresh = LLONG_MAX;
		config->offload_cgemm_thresh = LLONG_MAX;
                config->offload_zgemm_thresh = LLONG_MAX;
		return;
	}
	else if ( strcmp ( s_eng, "always" ) == 0 )
	{
		fprintf ( stdout, "Always attempting to offload.\n" );
		config->model = always;
		config->never_offload_dgemm = false;
		config->never_offload_sgemm = false;
		config->never_offload_zgemm = false;
                config->never_offload_cgemm = false;
		config->offload_sgemm_thresh = 0;
		config->offload_dgemm_thresh = 0;
		config->offload_cgemm_thresh = 0;
                config->offload_zgemm_thresh = 0;
		// still initialize rocBLAS handle
	}
	else if ( strcmp ( s_eng, "threshold" ) == 0 )
	{
		const char* s_sgemm = getenv ( "BLIS_OFFLOAD_SGEMM_THRESH" );
		const int64_t offload_after_s = ( s_sgemm == NULL ) ? LLONG_MAX : atol ( s_sgemm );
		config->model = threshold;
		config->offload_sgemm_thresh = offload_after_s;

		if ( offload_after_s == LLONG_MAX )
		{
			fprintf ( stdout, "Never offloading sgemms.\n" );
			config->never_offload_sgemm = true;
		}
		else
		{
			fprintf ( stdout, "Offloading all sgemms with at least M*N >= %ld\n", offload_after_s );
			config->never_offload_sgemm = false;
		}

		const char* s_dgemm = getenv ( "BLIS_OFFLOAD_DGEMM_THRESH" );
		const int64_t offload_after_d = ( s_dgemm == NULL ) ? LLONG_MAX : atol ( s_dgemm );
		config->offload_dgemm_thresh = offload_after_d;

		if ( offload_after_d == LLONG_MAX )
		{
			fprintf ( stdout, "Never offloading dgemms.\n" );
			config->never_offload_dgemm = true;
		}
		else
		{
			fprintf ( stdout, "Offloading all dgemms with at least M*N >= %ld\n", offload_after_d );
			config->never_offload_dgemm = false;
		}

		const char* s_cgemm = getenv ( "BLIS_OFFLOAD_CGEMM_THRESH" );
                const int64_t offload_after_c = ( s_sgemm == NULL ) ? LLONG_MAX : atol ( s_cgemm );
                config->offload_cgemm_thresh = offload_after_c;

                if ( offload_after_c == LLONG_MAX )
                {
                        fprintf ( stdout, "Never offloading cgemms.\n" );
                        config->never_offload_cgemm = true;
                }
                else
                {
                        fprintf ( stdout, "Offloading all cgemms with at least M*N >= %ld\n", offload_after_c );
                        config->never_offload_cgemm = false;
                }

                const char* s_zgemm = getenv ( "BLIS_OFFLOAD_ZGEMM_THRESH" );
                const int64_t offload_after_z = ( s_dgemm == NULL ) ? LLONG_MAX : atol ( s_zgemm );
                config->offload_zgemm_thresh = offload_after_z;

                if ( offload_after_z == LLONG_MAX )
                {
                        fprintf ( stdout, "Never offloading zgemms.\n" );
                        config->never_offload_zgemm = true;
                }
                else
                {
                        fprintf ( stdout, "Offloading all zgemms with at least M*N >= %ld\n", offload_after_z );
                        config->never_offload_zgemm = false;
                }

		// still initialize rocBLAS handle
	}
        else if ( strcmp ( s_eng, "pm1" ) == 0 )
        {
		fprintf ( stdout, "Using PM1 to decide offload.\n" );
		config->model = pm1;
		config->never_offload_dgemm = false;
		config->never_offload_sgemm = false;
		config->never_offload_zgemm = false;
		config->never_offload_cgemm = false;

		char* s_pm1_file = getenv ( "BLIS_PM1_PARAMS_FILE" );
		s_pm1_file = ( s_pm1_file == NULL ) ? "pm1_params.txt" : s_pm1_file;

		FILE *fp = NULL;
		size_t file_size = 0;
		char *buff = NULL;

		fp = fopen ( s_pm1_file , "rb" );
		if ( !fp )
		{
			fprintf ( stderr, "BLIS offload: Failed to open file %s\n", s_pm1_file );
			exit ( 1 );
		}

		fseek ( fp , 0L , SEEK_END);
		file_size = ftell( fp );
		rewind ( fp );

		buff = calloc( file_size + 1, sizeof( char ) );
		if ( !buff )
		{
			fprintf ( stderr, "BLIS offload: calloc failed\n" );
			fclose ( fp );
			exit ( 1 );
		}

		if ( 1 != fread( buff , file_size, 1 , fp) )
		{
			fprintf ( stderr, "BLIS offload: PM1 file read failed.\n" );
			fclose ( fp );
			free ( buff );
			exit ( 1 );
		}

		fclose ( fp );

		// parse json
		const int json_status =  json_read_object ( buff, json_attrs_bli_off, NULL );
		if ( json_status != 0 )
			fprintf(stderr, "BLIS offload: illegal status of parsing PM1 parameters: %d\n", json_status );

		free ( buff );
		// still initialize rocBLAS handle
	}
	else
	{
		fprintf ( stderr, "Unknown BLIS_OFFLOAD selection: %s . Offloading never.\n", s_eng );
		config->never_offload_dgemm = true;
		config->never_offload_sgemm = true;
		config->never_offload_zgemm = true;
                config->never_offload_cgemm = true;
		config->offload_sgemm_thresh = LLONG_MAX;
		config->offload_dgemm_thresh = LLONG_MAX;
		config->offload_cgemm_thresh = LLONG_MAX;
                config->offload_zgemm_thresh = LLONG_MAX;
		return;
	}

	const rocblas_status stat = rocblas_create_handle ( & ( config->rocblas ) );
	if ( stat != rocblas_status_success )
	{
		fprintf ( stderr, "Couldn't create rocBLAS handle w/ error %d\n", stat );
	}
	const rocblas_status stat_p = rocblas_set_pointer_mode ( config->rocblas,
	                              rocblas_pointer_mode_host );
	if ( stat_p != rocblas_status_success )
	{
		fprintf ( stderr, "Couldn't set rocBLAS pointer mode to host w/ error %d\n", stat );
	}
}

void bli_offloader_finalize ( void )
{
	bli_offloader_finalize_rntm_from_env ( &global_rntm );
}

void bli_offloader_finalize_rntm_from_env ( rntm_t* rntm )
{
	if ( rntm->offloader_state->rocblas != NULL )
	{
		// just destroy rocblas handle
		const rocblas_status stat = rocblas_destroy_handle ( rntm->offloader_state->rocblas );
		if ( stat != rocblas_status_success )
		{
			fprintf ( stderr, "Couldn't destroy rocBLAS handle w/ error %d\n", stat );
		}
	}

	// free struct itself
	free ( rntm->offloader_state );
}

bool bli_do_offload_gemmex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{
	return bli_do_offload_gemmex_rntm_from_env ( &global_rntm, alpha, a, b, beta, c );
}

bool bli_do_offload_gemmex_rntm_from_env
     (
       rntm_t* rntm,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{

	bli_offload_t* config = rntm->offloader_state;

        // figure out data type and whether we offload for it is enabled at all
        const bool is_float_c = bli_obj_is_float ( c );
        const bool is_double_c = bli_obj_is_double ( c );
	const bool is_scmpl_c = bli_obj_is_scomplex ( c );
	const bool is_dcmpl_c = bli_obj_is_dcomplex ( c );
	const bool is_compl_c = is_scmpl_c || is_dcmpl_c;
        if ( is_float_c && config->never_offload_sgemm )
        {
                return false;
        }
        else if ( is_double_c && config->never_offload_dgemm )
        {
                return false;
        }
        else if ( is_scmpl_c && config->never_offload_cgemm )
        {
                return false;
        }
        else if ( is_dcmpl_c && config->never_offload_zgemm )
        {
                return false;
        }

	// figure out if C is integer and reject (for now)
	// NOTE: rocBLAS supports f16, f16 cmpl, f32, f32 cmpl, f64, f64 cmpl, i8, u8, i32,
	//       i32 cmpl, u32 compl, bf16, bf16 cmpl as data type settings
	//       (not in all combinations)
	if ( bli_obj_is_int ( a ) ||  bli_obj_is_int ( b ) || bli_obj_is_int ( c ) )
	{
		return false;
	}

	const inc_t rs_a = bli_obj_row_stride ( a );
	const inc_t rs_b = bli_obj_row_stride ( b );
	const inc_t rs_c = bli_obj_row_stride ( c );
	// do not offload if any row stride is != 1 (as rocBLAS only supports col strides)
	if ( rs_a != 1 || rs_b != 1 || rs_c != 1 )
	{
		return false;
	}

	if ( config->model == always ) return true;
	else if ( config->model == never ) return false;
	else if ( config->model == threshold )
        {
		// figure out if the M*N*K effort is above or below the data type specific cutoff
		const dim_t m_c = bli_obj_length ( c );
		const dim_t n_c = bli_obj_width ( c );
		const dim_t k_a = bli_obj_has_trans ( a ) ? bli_obj_width ( a ) : bli_obj_length ( a );
		const size_t mul = m_c * n_c * k_a;

		if ( !is_compl_c )
		{
			const int64_t thresh = ( is_float_c ) ? config->offload_sgemm_thresh : config->offload_dgemm_thresh;
			return ( mul >= thresh );
		}
		else
		{
			// make sure we're not conjugate AND not transpose
			if ( bli_obj_has_conj( a ) && !bli_obj_has_trans( a ) ) return false;
			if ( bli_obj_has_conj( b ) && !bli_obj_has_trans( b ) ) return false;

			const int64_t thresh = ( is_scmpl_c ) ? config->offload_cgemm_thresh : config->offload_zgemm_thresh;
			return ( mul >= thresh );
		}
	}
	else if ( config->model == pm1 )
	{
		// figure out the M*N*K effort
		const dim_t m_c = bli_obj_length ( c );
		const dim_t n_c = bli_obj_width ( c );
		const dim_t k_a = bli_obj_has_trans ( a ) ? bli_obj_width ( a ) : bli_obj_length ( a );
                const size_t mnk = m_c * n_c * k_a;

		if ( is_compl_c )
                {
                        // make sure we're not conjugate AND not transpose
                        if ( bli_obj_has_conj( a ) && !bli_obj_has_trans( a ) ) return false;
                        if ( bli_obj_has_conj( b ) && !bli_obj_has_trans( b ) ) return false;
                }

		double mem_copy_cost_to_cpu = 0.0;
		double mem_copy_cost_to_acc = 0.0;
		if ( !bli_off_mem_trans )
		{
			void *A = bli_obj_buffer_at_off ( a ); // pointer to elements of Matrix A
		        void *B = bli_obj_buffer_at_off ( b ); // pointer to elements of Matrix B
			void *C = bli_obj_buffer_at_off ( c ); // pointer to elements of Matrix C

			const inc_t lda = bli_obj_col_stride ( a );
			const inc_t ldb = bli_obj_col_stride ( b );
			const inc_t ldc = bli_obj_col_stride ( c );
			const dim_t n_a = bli_obj_width ( a );
			const dim_t n_b = bli_obj_width ( b );
			const size_t buff_size_a = lda * n_a * bli_obj_elem_size ( a );
        		const size_t buff_size_b = ldb * n_b * bli_obj_elem_size ( b );
        		const size_t buff_size_c = ldc * n_c * bli_obj_elem_size ( c );

			// actually inspect all the pointers for their location
			hipPointerAttribute_t attr;
			bool a_on_dev = false;
	                const hipError_t err_insp_a = hipPointerGetAttributes(&attr, A);
        	        if ( err_insp_a == hipSuccess )
                	{
                        	a_on_dev = ( attr.memoryType == hipMemoryTypeDevice );
                	}
			bool b_on_dev = false;
                	const hipError_t err_insp_b = hipPointerGetAttributes(&attr, B);
                	if ( err_insp_b == hipSuccess )
                	{
                        	b_on_dev = ( attr.memoryType == hipMemoryTypeDevice );
                	}
			bool c_on_dev = false;
                	const hipError_t err_insp_c = hipPointerGetAttributes(&attr, C);
                	if ( err_insp_c == hipSuccess )
                	{
                        	c_on_dev = ( attr.memoryType == hipMemoryTypeDevice );
                	}

			// compute copy cost to or from device based on whether a copy is necessary
			double cost_a_to_cpu = 0.0, cost_a_to_acc = 0.0, cost_b_to_cpu = 0.0,
			       cost_b_to_acc = 0.0, cost_c_to_cpu = 0.0, cost_c_to_acc = 0.0;

			if ( a_on_dev )
			{
				cost_a_to_cpu = bli_off_pm1_mem_cost ( buff_size_a ); 
			}
			else
			{
				cost_a_to_acc = bli_off_pm1_mem_cost ( buff_size_a );
			}
			if ( b_on_dev )
                        {
                                cost_b_to_cpu = bli_off_pm1_mem_cost ( buff_size_b );
                        }
                        else
                        {
                                cost_b_to_acc = bli_off_pm1_mem_cost ( buff_size_b );
			}
			if ( c_on_dev )
                        {
                                cost_c_to_cpu = bli_off_pm1_mem_cost ( buff_size_c );
                        }
                        else
                        {
                                cost_c_to_acc = bli_off_pm1_mem_cost ( buff_size_c );
			}

			mem_copy_cost_to_cpu = cost_a_to_cpu + cost_b_to_cpu + cost_c_to_cpu;
			mem_copy_cost_to_acc = cost_a_to_acc + cost_b_to_acc + cost_c_to_acc;
		}

		// gemm cost
		const bool is_squarish_gemm = bli_is_squarish ( m_c, n_c, k_a);
		// [s,d,c,z]
		double a_cpu = 0.0, b_cpu = 0.0, a_acc = 0.0, b_acc = 0.0;
		if ( is_float_c )
		{
			if ( is_squarish_gemm )
			{
				a_cpu = bli_off_sgemm_sq[BLI_OFF_CPU_A];
				b_cpu = bli_off_sgemm_sq[BLI_OFF_CPU_B];
				a_acc = bli_off_sgemm_sq[BLI_OFF_ACC_A];
				b_acc = bli_off_sgemm_sq[BLI_OFF_ACC_B];
			}
			else
			{
				a_cpu = bli_off_sgemm[BLI_OFF_CPU_A];
                                b_cpu = bli_off_sgemm[BLI_OFF_CPU_B];
                                a_acc = bli_off_sgemm[BLI_OFF_ACC_A];
                                b_acc = bli_off_sgemm[BLI_OFF_ACC_B];
			}
		}
		else if ( is_double_c )
		{
			if ( is_squarish_gemm )
                        {
                                a_cpu = bli_off_dgemm_sq[BLI_OFF_CPU_A];
                                b_cpu = bli_off_dgemm_sq[BLI_OFF_CPU_B];
                                a_acc = bli_off_dgemm_sq[BLI_OFF_ACC_A];
                                b_acc = bli_off_dgemm_sq[BLI_OFF_ACC_B];
                        }
                        else
                        {
                                a_cpu = bli_off_dgemm[BLI_OFF_CPU_A];
                                b_cpu = bli_off_dgemm[BLI_OFF_CPU_B];
                                a_acc = bli_off_dgemm[BLI_OFF_ACC_A];
                                b_acc = bli_off_dgemm[BLI_OFF_ACC_B];
                        }
		}
		else if ( is_scmpl_c )
		{
			if ( is_squarish_gemm )
                        {
                                a_cpu = bli_off_cgemm_sq[BLI_OFF_CPU_A];
                                b_cpu = bli_off_cgemm_sq[BLI_OFF_CPU_B];
                                a_acc = bli_off_cgemm_sq[BLI_OFF_ACC_A];
                                b_acc = bli_off_cgemm_sq[BLI_OFF_ACC_B];
                        }
                        else
                        {
                                a_cpu = bli_off_cgemm[BLI_OFF_CPU_A];
                                b_cpu = bli_off_cgemm[BLI_OFF_CPU_B];
                                a_acc = bli_off_cgemm[BLI_OFF_ACC_A];
                                b_acc = bli_off_cgemm[BLI_OFF_ACC_B];
                        }
		}
		else if ( is_dcmpl_c )
		{
			if ( is_squarish_gemm )
                        {
                                a_cpu = bli_off_zgemm_sq[BLI_OFF_CPU_A];
                                b_cpu = bli_off_zgemm_sq[BLI_OFF_CPU_B];
                                a_acc = bli_off_zgemm_sq[BLI_OFF_ACC_A];
                                b_acc = bli_off_zgemm_sq[BLI_OFF_ACC_B];
                        }
                        else
                        {
                                a_cpu = bli_off_zgemm[BLI_OFF_CPU_A];
                                b_cpu = bli_off_zgemm[BLI_OFF_CPU_B];
                                a_acc = bli_off_zgemm[BLI_OFF_ACC_A];
                                b_acc = bli_off_zgemm[BLI_OFF_ACC_B];
                        }
		}
		else
		{
			fprintf ( stderr, "Unknown case for PM1 gemm model.\n" );
			return false;
		}
		
		const double gemm_cost_cpu = bli_off_pm1_gemm_cost ( mnk, a_cpu, b_cpu);
		const double gemm_cost_acc = bli_off_pm1_gemm_cost ( mnk, a_acc, b_acc);

		return ( mem_copy_cost_to_acc + gemm_cost_acc ) < ( mem_copy_cost_to_cpu + gemm_cost_cpu );
	}

	// default false
	return false;
}


err_t bli_offload_gemmex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{
	return bli_offload_gemmex_rntm_from_env ( &global_rntm, alpha, a, b, beta, c );

}

err_t bli_offload_gemmex_rntm_from_env
     (
       rntm_t* rntm,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{

	bli_offload_t* config = rntm->offloader_state;

	// at this point we will offload - no checking with model or for compat
	// any error belongs to the user now

        // check for float
	const bool is_float_a = bli_obj_is_float ( a );
	const bool is_float_b = bli_obj_is_float ( b );
	const bool is_float_c = bli_obj_is_float ( c );

	// are any of the matrices complex
        const bool is_compl_a = bli_obj_is_complex ( a );
        const bool is_compl_b = bli_obj_is_complex ( b );
	const bool is_compl_c = bli_obj_is_complex ( c );

	const inc_t lda = bli_obj_col_stride ( a );
	const inc_t ldb = bli_obj_col_stride ( b );
	const inc_t ldc = bli_obj_col_stride ( c );
	const dim_t m_a = bli_obj_length ( a );
	const dim_t n_a = bli_obj_width ( a );
	const dim_t n_b = bli_obj_width ( b );
	const dim_t m_c = bli_obj_length ( c );
	const dim_t n_c = bli_obj_width ( c );

	void *A = bli_obj_buffer_at_off ( a ); // pointer to elements of Matrix A
	void *B = bli_obj_buffer_at_off ( b ); // pointer to elements of Matrix B
	void *C = bli_obj_buffer_at_off ( c ); // pointer to elements of Matrix C

	const bool is_trans_a = bli_obj_has_trans ( a );
	const bool is_trans_b = bli_obj_has_trans ( b );

	const size_t buff_size_a = lda * n_a * bli_obj_elem_size ( a );
	const size_t buff_size_b = ldb * n_b * bli_obj_elem_size ( b );
	const size_t buff_size_c = ldc * n_c * bli_obj_elem_size ( c );

	bool copy_a = false, copy_b = false, copy_c = false;
	if ( config->model != pm1 || bli_off_mem_trans )
        {	
		// inspect pointers for memory location of buffers
		hipPointerAttribute_t attr;
		const hipError_t err_insp_a = hipPointerGetAttributes(&attr, A);
		if ( err_insp_a == hipSuccess )
		{
			copy_a = ( attr.memoryType != hipMemoryTypeDevice );
	   	}
		else
		{
			// failure to inspect may happen for host pointers
			copy_a = true;
		}
		const hipError_t err_insp_b = hipPointerGetAttributes(&attr, B);
		if ( err_insp_b == hipSuccess )
        	{
                	copy_b = ( attr.memoryType != hipMemoryTypeDevice );
        	}
		else
                {
			// failure to inspect may happen for host pointers
                        copy_b = true;
                }
		const hipError_t err_insp_c = hipPointerGetAttributes(&attr, C);
        	if ( err_insp_c == hipSuccess )
        	{
                	copy_c = ( attr.memoryType != hipMemoryTypeDevice );
        	}
		else
                {
			// failure to inspect may happen for host pointers
                        copy_c = true;
                }
	}

	// if applicable: allocate buffers on device and copy data
	// note: we cannot assume the CPU buffers to be pinned and hence most likely the copies will be synchronous
	void* dev_buff_a;
        void* dev_buff_b;
	void* dev_buff_c;

        hipStream_t stream;
        rocblas_get_stream( config->rocblas, &stream );

	if ( copy_a )
	{
		const hipError_t err_a = hipMalloc ( &dev_buff_a, buff_size_a );
		if ( err_a != hipSuccess )
		{
			fprintf ( stderr, "Failure to allocate device buffer A of size %ld: %d\n", buff_size_a, err_a );
			return BLIS_FAILURE;
		}
		const hipError_t err_cpa = hipMemcpy ( dev_buff_a, A, buff_size_a, hipMemcpyHostToDevice );
        	if ( err_cpa != hipSuccess )
        	{
                	fprintf ( stderr, "Failure to hipMemcpy A to device: %d\n", err_cpa );
                	return BLIS_FAILURE;
        	}
	}
	else
	{
		dev_buff_a = A;
	}

	if ( copy_b )
	{
		const hipError_t err_b = hipMalloc ( &dev_buff_b, buff_size_b );
		if ( err_b != hipSuccess )
		{
			fprintf ( stderr, "Failure to allocate device buffer B of size %ld: %d\n", buff_size_b, err_b );
			return BLIS_FAILURE;
		}
		const hipError_t err_cpb = hipMemcpy ( dev_buff_b, B, buff_size_b, hipMemcpyHostToDevice );
	        if ( err_cpb != hipSuccess )
        	{
                	fprintf ( stderr, "Failure to hipMemcpy B to device: %d\n", err_cpb );
			return BLIS_FAILURE;
        	}

	}
	else
	{
		dev_buff_b = B;
	}

	if ( copy_c )
	{
		const hipError_t err_c = hipMalloc ( &dev_buff_c, buff_size_c );
		if ( err_c != hipSuccess )
		{
			fprintf ( stderr, "Failure to allocate device buffer C of size %ld: %d\n", buff_size_c, err_c );
			return BLIS_FAILURE;
		}

		// is beta zero?
		const bool is_beta_non_zero = !bli_obj_equals ( beta, &BLIS_ZERO );

		if ( is_beta_non_zero || ldc != m_c ) // only if the result buffer is m*n sized AND beta == 0.0 we can eschew the copy
		{
			const hipError_t err_cpc = hipMemcpy ( dev_buff_c, C, buff_size_c, hipMemcpyHostToDevice );
			if ( err_cpc != hipSuccess )
			{
				fprintf ( stderr, "Failure to hipMemcpy C to device: %d\n", err_cpc );
				return BLIS_FAILURE;
			}
		}
	}

	// call rocblas
	rocblas_operation trans_a = is_trans_a ? rocblas_operation_transpose : rocblas_operation_none;
	rocblas_operation trans_b = is_trans_b ? rocblas_operation_transpose : rocblas_operation_none;
        if ( is_compl_a && bli_obj_has_conj( a ) )
		trans_a = rocblas_operation_conjugate_transpose;
	if ( is_compl_b && bli_obj_has_conj( b ) )
                trans_b = rocblas_operation_conjugate_transpose;

	rocblas_datatype a_type;
	rocblas_datatype b_type;
	rocblas_datatype c_type;
	if ( is_compl_a )
		a_type = ( bli_obj_is_scomplex( a ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;
	else
		a_type = ( is_float_a ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;
	if ( is_compl_b )
                b_type = ( bli_obj_is_scomplex( b ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;
        else
                b_type = ( is_float_b ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;
	if ( is_compl_c )
                c_type = ( bli_obj_is_scomplex( c ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;
        else
                c_type = ( is_float_c ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;

	rocblas_datatype compute_type;
	if ( !is_compl_a && !is_compl_b && !is_compl_c)
		compute_type = ( is_float_a && is_float_b && is_float_c ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;
	else
		compute_type = ( bli_obj_is_scomplex( a ) && bli_obj_is_scomplex( b ) && bli_obj_is_scomplex( c ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;

	const num_t    dt_exec   = bli_obj_dt ( c );
	void* restrict alpha_f = bli_obj_buffer_for_1x1 ( dt_exec, alpha );
	void* restrict beta_f  = bli_obj_buffer_for_1x1 ( dt_exec, beta );


	const size_t k = is_trans_a ? m_a : n_a;
	const rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
	const int32_t solution_index = 0;
	const uint32_t flags = 0;
	const rocblas_status roc_err = rocblas_gemm_ex ( config->rocblas,
	                               trans_a,
	                               trans_b,
	                               m_c,
	                               n_c,
	                               k,
	                               alpha_f,
	                               dev_buff_a,
	                               a_type,
	                               lda,
	                               dev_buff_b,
	                               b_type,
	                               ldb,
	                               beta_f,
	                               dev_buff_c,
	                               c_type,
	                               ldc,
	                               dev_buff_c,
	                               c_type,
	                               ldc,
	                               compute_type,
	                               algo,
	                               solution_index,
	                               flags );
	if ( roc_err != rocblas_status_success )
	{
		fprintf ( stderr, "Failure to call rocblas_dgemm: %d\n", roc_err );
		return BLIS_FAILURE;
	}

	// if applicable: free intermediate buffers
        if ( copy_a )
        {
                const hipError_t err_fa = hipFree ( dev_buff_a );
                if ( err_fa != hipSuccess )
                {
                        fprintf ( stderr, "Failure to free device buffer A: %d\n", err_fa );
                        return BLIS_FAILURE;
                }
        }
        if ( copy_b )
        {
                const hipError_t err_fb = hipFree ( dev_buff_b );
                if ( err_fb != hipSuccess )
                {
                        fprintf ( stderr, "Failure to free device buffer B: %d\n", err_fb );
                        return BLIS_FAILURE;
                }
        }

	if ( copy_c )
	{
		// copy result back synchronously
		const hipError_t err_cpr = hipMemcpy ( C, dev_buff_c, buff_size_c, hipMemcpyDeviceToHost );
		if ( err_cpr != hipSuccess )
		{
			fprintf ( stderr, "Failure to hipMemcpy C from device: %d\n", err_cpr );
			return BLIS_FAILURE;
		}
		// free
		const hipError_t err_fc = hipFree ( dev_buff_c );
	        if ( err_fc != hipSuccess )
        	{
                	fprintf ( stderr, "Failure to free device buffer C: %d\n", err_fc );
                	return BLIS_FAILURE;
        	}
	}
	else
	{
		// only synchronize on the rocBLAS stream to ensure data correctness
		hipStreamSynchronize( stream );
	}

	return BLIS_SUCCESS;
}

static inline bool bli_is_squarish ( const dim_t m, const dim_t n, const dim_t k )
{
        const dim_t max_mn = bli_max ( m, n );
        const dim_t max_mnk = bli_max ( max_mn, k );
        const dim_t min_mn = bli_min ( m, n );
        const dim_t min_mnk = bli_min ( min_mn, k );

        return ( max_mnk <= 2 * min_mnk );
}

static inline double bli_off_pm1_mem_cost ( const size_t length )
{
        return bli_off_a_mem * length + bli_off_b_mem;
}

static inline double bli_off_pm1_gemm_cost ( const size_t mnk, const double a, const double b )
{
        return a * mnk + b;
}

#endif
#include <stdint.h>
static __global__ void _kernel_0(double BETA, int32_t M, int32_t *b_col_idx_gpu, double *b_partial_res_gpu, int32_t *b_row_start_gpu, double *b_spmv_gpu, double *b_values_gpu, double *b_w_gpu, double *b_x_gpu, double *b_y_gpu)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	b_w_gpu[(0 + (((512 * __bx__) + __tx__) * 1))] = (b_x_gpu[(0 + (((512 * __bx__) + __tx__) * 1))] + (BETA * b_y_gpu[(0 + (((512 * __bx__) + __tx__) * 1))]));
	__syncthreads();
	const int32_t b0 = b_row_start_gpu[(0 + (((512 * __bx__) + __tx__) * 1))];
	const int32_t b1 = b_row_start_gpu[(0 + ((((512 * __bx__) + __tx__) + 1) * 1))];
	for (int32_t c5 = b0; (c5 < b1); (c5 += 1))
	{
		const int32_t t = b_col_idx_gpu[(0 + (1 * ((int32_t) c5)))];
		b_spmv_gpu[(0 + (((512 * __bx__) + __tx__) * 1))] = (b_spmv_gpu[(0 + (((512 * __bx__) + __tx__) * 1))] + (b_values_gpu[(0 + (c5 * 1))] * b_w_gpu[(0 + (t * 1))]));
	};
	__syncthreads();
	__shared__ double sdata[512];
	sdata[(0 + (__tx__ * 1))] = 0;
	sdata[(0 + (__tx__ * 1))] = (b_spmv_gpu[(0 + (((512 * __bx__) + __tx__) * 1))] * b_w_gpu[(0 + (((512 * __bx__) + __tx__) * 1))]);
	if ((__tx__ <= 255))
	{
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 256) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 127))
	{
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 128) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 63))
	{
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 64) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 31))
	{
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 32) * 1))]);
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 16) * 1))]);
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 8) * 1))]);
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 4) * 1))]);
		sdata[(0 + (__tx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 2) * 1))]);
		b_partial_res_gpu[(0 + (__bx__ * 1))] = (sdata[(0 + (__tx__ * 1))] + sdata[(0 + ((__tx__ + 1) * 1))]);
	};
};
extern "C" int32_t _kernel_0_wrapper(double BETA, int32_t M, int32_t *b_col_idx_gpu, double *b_partial_res_gpu, int32_t *b_row_start_gpu, double *b_spmv_gpu, double *b_values_gpu, double *b_w_gpu, double *b_x_gpu, double *b_y_gpu)
{
	{
		dim3 blocks(((M / 512) + 1), 1, 1);
		dim3 threads((511 + 1), 1, 1);
		_kernel_0<<<blocks, threads>>>(BETA, M, b_col_idx_gpu, b_partial_res_gpu, b_row_start_gpu, b_spmv_gpu, b_values_gpu, b_w_gpu, b_x_gpu, b_y_gpu);
	};
	return 0;
}
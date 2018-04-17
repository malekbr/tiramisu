#include <stdint.h>
static __global__ void _kernel_0(int32_t M, int32_t MR, double *reduced_1_gpu, double *x_gpu, double *y_gpu)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	__shared__ double dot_gpu[512];
	dot_gpu[(0 + (__tx__ * 1))] = ((((int32_t) ((512 * __bx__) + __tx__)) < M) ? (x_gpu[(0 + (((512 * __bx__) + __tx__) * 1))] * y_gpu[(0 + (((512 * __bx__) + __tx__) * 1))]) : 0);
	if ((__tx__ <= 255))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 256) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 127))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 128) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 63))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 64) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 31))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 32) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 15))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 16) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 7))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 8) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 3))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 4) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 1))
	{
		dot_gpu[(0 + (__tx__ * 1))] = (dot_gpu[(0 + (__tx__ * 1))] + dot_gpu[(0 + ((__tx__ + 2) * 1))]);
	};
	__syncthreads();
	if ((__tx__ == 0))
	{
		dot_gpu[(0 + (0 * 1))] = (dot_gpu[(0 + (0 * 1))] + dot_gpu[(0 + (1 * 1))]);
	};
	__syncthreads();
	if ((__tx__ == 0))
	{
		reduced_1_gpu[(0 + (__bx__ * 1))] = dot_gpu[(0 + (0 * 1))];
	};
};
extern "C" int32_t _kernel_0_wrapper(int32_t M, int32_t MR, double *reduced_1_gpu, double *x_gpu, double *y_gpu)
{
	{
		dim3 blocks((((M - 1) / 512) + 1), 1, 1);
		dim3 threads((511 + 1), 1, 1);
		_kernel_0<<<blocks, threads>>>(M, MR, reduced_1_gpu, x_gpu, y_gpu);
	};
	return 0;
};
static __global__ void _kernel_1(int32_t MR, int32_t MR2, double *reduced_1_gpu, double *reduced_2_gpu)
{
	const int32_t __bx__ = (blockIdx.x + 0);
	const int32_t __tx__ = (threadIdx.x + 0);
	__shared__ double sum_gpu[512];
	sum_gpu[(0 + (__tx__ * 1))] = ((((int32_t) ((512 * __bx__) + __tx__)) < MR) ? reduced_1_gpu[(0 + (((512 * __bx__) + __tx__) * 1))] : 0);
	if ((__tx__ <= 255))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 256) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 127))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 128) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 63))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 64) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 31))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 32) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 15))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 16) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 7))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 8) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 3))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 4) * 1))]);
	};
	__syncthreads();
	if ((__tx__ <= 1))
	{
		sum_gpu[(0 + (__tx__ * 1))] = (sum_gpu[(0 + (__tx__ * 1))] + sum_gpu[(0 + ((__tx__ + 2) * 1))]);
	};
	__syncthreads();
	if ((__tx__ == 0))
	{
		sum_gpu[(0 + (0 * 1))] = (sum_gpu[(0 + (0 * 1))] + sum_gpu[(0 + (1 * 1))]);
	};
	__syncthreads();
	if ((__tx__ == 0))
	{
		reduced_2_gpu[(0 + (__bx__ * 1))] = sum_gpu[(0 + (0 * 1))];
	};
};
extern "C" int32_t* _kernel_1_wrapper(int32_t MR, int32_t MR2, double *reduced_1_gpu, double *reduced_2_gpu)
{
	{
		dim3 blocks((((MR - 1) / 512) + 1), 1, 1);
		dim3 threads((511 + 1), 1, 1);
		_kernel_1<<<blocks, threads>>>(MR, MR2, reduced_1_gpu, reduced_2_gpu);
	};
	return 0;
}
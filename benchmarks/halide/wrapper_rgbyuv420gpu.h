#ifndef HALIDE__build___wrapper_rgbyuv420gpu_o_h
#define HALIDE__build___wrapper_rgbyuv420gpu_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int rgbyuv420gpu_tiramisu(halide_buffer_t *, halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_y_buffer, halide_buffer_t *_b_output_u_buffer, halide_buffer_t *_b_output_v_buffer);
int rgbyuv420gpu_tiramisu_argv(void **args);
int rgbyuv420gpu_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_y_buffer, halide_buffer_t *_b_output_u_buffer, halide_buffer_t *_b_output_v_buffer);
int rgbyuv420gpu_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *rgbyuv420gpu_tiramisu_metadata();
const struct halide_filter_metadata_t *rgbyuv420gpu_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif

#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include "benchmarks.h"

#include <string.h>
#include <Halide.h>


using namespace tiramisu;

#define THREADS 32
#define B0 64
#define B1 8
#define B2 16
#define PARTITIONS (67108864/THREADS)

#define USE_GPU true


int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function cg("cg");


    // Inputs
    computation SIZES("[M]->{SIZES[0]}", tiramisu::expr(), false, p_int32, &cg);
    computation x("[M]->{x[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &cg);
    computation y("[M]->{y[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &cg);
    computation alpha("[M]->{alpha[0]}", tiramisu::expr(), false, p_float64, &cg);
    computation beta("[M]->{beta[0]}", tiramisu::expr(), false, p_float64, &cg);

    constant M_CST("M", SIZES(0), p_int32, true, NULL, 0, &cg);
    constant ALPHA("ALPHA", alpha(0), p_float64, true, NULL, 0, &cg);
    constant BETA("BETA", beta(0), p_float64, true, NULL, 0, &cg);

    tiramisu::var j("j"), j0("j0"), j1("j1");
    computation w("[M]->{w[j]: 0<=j<M}", x(j) + (expr)BETA*y(j), true, p_float64, &cg);


    computation c_row_start("[M]->{c_row_start[i]: 0<=i<M}", tiramisu::expr(), false, p_int32, &cg);
    computation c_col_idx("[b0,b1]->{c_col_idx[j]: b0<=j<b1}", tiramisu::expr(), false, p_int32, &cg);
    computation c_values("[b0,b1]->{c_values[j]: b0<=j<b1}", tiramisu::expr(), false, p_float64, &cg);
    computation c_spmv("[M,b0,b1]->{c_spmv[i,j]: 0<=i<M and b0<=j<b1}", tiramisu::expr(), true, p_float64, &cg);
    computation c_spmv_wrapper("[M,b0,b1]->{c_spmv_wrapper[i]: 0<=i<M}", tiramisu::expr(), false, p_float64, &cg);

#ifdef USE_GPU
    c_spmv.add_associated_let_stmt("t", c_col_idx(var("j")));
#else
    c_spmv.add_associated_let_stmt("t", c_col_idx(var("c7")));
#endif
    constant b1("b1", c_row_start(var("i") + 1), p_int32, false, &c_spmv, 0, &cg);
    constant b0("b0", c_row_start(var("i")), p_int32, false, &b1, 0, &cg);

    expr e_y = c_spmv(var("i"), var("j")) + c_values(var("j")) * w(var("t")); //(x(var("t")) + beta(0)*y(var("t")));
    c_spmv.set_expression(e_y);


    // dot
#ifdef USE_GPU
    computation sync_stage_1{"[M]->{sync_stage_1[j]: 0 <= j < ceil(M / 512) * 512}", tiramisu::sync{}, true, p_none, &cg};
    computation sync_stage_2{"[M]->{sync_stage_2[j]: 0 <= j < ceil(M / 512) * 512}", tiramisu::sync{}, true, p_none, &cg};
    computation sdata_alloc("[M] -> {sdata_alloc[j]: 0 <= j < ceil(M / 512) * 512}", expr{o_allocate, "sdata"}, true, p_none, &cg);
    computation sdata_init("[M] -> {sdata_init[j]: 0 <= j < ceil(M / 512) * 512}", 0., true, p_float64, &cg);
    computation mul("[M]->{ mul[j]: 0<=j<M}", c_spmv_wrapper(j)*w(j), true, p_float64, &cg);
    computation reduction256{"[M]->{reduction256[j]: 0 <= j < M and j % 512 < 256}", mul(j) + mul(j + 256), true, p_float64, &cg};
    computation sync256{"[M]->{sync256[j]: 0 <= j < ceil(M / 512) * 512}", tiramisu::sync{}, true, p_none, &cg};
    computation reduction128{"[M]->{reduction128[j]: 0 <= j < M and j % 512 < 128}", mul(j) + mul(j + 128), true, p_float64, &cg};
    computation sync128{"[M]->{sync128[j]: 0 <= j < ceil(M / 512) * 512}", tiramisu::sync{}, true, p_none, &cg};
    computation reduction64{"[M]->{reduction64[j]: 0 <= j < M and j % 512 < 64}", mul(j) + mul(j + 64), true, p_float64, &cg};
    computation sync64{"[M]->{sync64[j]: 0 <= j < ceil(M / 512) * 512}", tiramisu::sync{}, true, p_none, &cg};
    // at the warp level, no need to sync
    computation reduction32{"[M]->{reduction32[j]: 0 <= j < M and j % 512 < 32}", mul(j) + mul(j + 32), true, p_float64, &cg};
    computation reduction16{"[M]->{reduction16[j]: 0 <= j < M and j % 512 < 32}", mul(j) + mul(j + 16), true, p_float64, &cg};
    computation reduction8 {"[M]->{reduction8 [j]: 0 <= j < M and j % 512 < 32}", mul(j) + mul(j +  8), true, p_float64, &cg};
    computation reduction4 {"[M]->{reduction4 [j]: 0 <= j < M and j % 512 < 32}", mul(j) + mul(j +  4), true, p_float64, &cg};
    computation reduction2 {"[M]->{reduction2 [j]: 0 <= j < M and j % 512 < 32}", mul(j) + mul(j +  2), true, p_float64, &cg};
    computation reduction1 {"[M]->{reduction1 [j]: 0 <= j < M and j % 512 < 32}", mul(j) + mul(j +  1), true, p_float64, &cg};
    computation reduced{"[M] -> {reduced[j]: 0 <= j < ceil(M / 512)}", expr{}, false, p_float64, &cg};
    computation res_init{"[M] -> {res_init[0]}", 0., true, p_float64, &cg};
    computation res{"[M] -> {res[j]: 1 <= j < ceil(M / 512)}", res_init(0) + reduced(j), true, p_float64, &cg};
#else
    computation res_alloc("[M]->{res_alloc[-10]}", tiramisu::expr(tiramisu::o_allocate, "b_res"), true, p_none, &cg);
    computation  res_init("[M]->{ res_init[t]: 0<=t<(M/"+std::to_string(PARTITIONS)+")}", tiramisu::expr((double) 0), true, p_float64, &cg);
    computation mul_alloc("[M]->{mul_alloc[j]: 0<=j<(M/"+std::to_string(PARTITIONS)+")}", tiramisu::expr(tiramisu::o_allocate, "b_mul"), true, p_float64, &cg);
    computation       mul("[M]->{ mul[j]: 0<=j<M}", c_spmv_wrapper(j)*w(j), true, p_float64, &cg);
    computation       res("[M]->{ res[j]: 0<=j<M}", tiramisu::expr(), true, p_float64, &cg);
    res.set_expression(res(j) + mul(j));
    computation res_global("[M]->{res_global[t]: 0<=t<(M/"+std::to_string(PARTITIONS)+")}", tiramisu::expr(),    true, p_float64, &cg);
    res_global.set_expression(res_global(var("t")) + res_init(var("t")));
#endif


    cg.set_context_set("[M,b0,b1]->{: M>0 and M%"+std::to_string(PARTITIONS)+"=0 and b0>0 and b1>0 and b1>b0}");

    // -----------------------------------------------------------------
    // Layer II
    // ----------------------------------------------------------------- 
#ifdef USE_GPU
    w.split(j, 512, j0, j1);
    w.tag_gpu_level(j0, j1);
    sync_stage_1.split(j, 512, j0, j1);
    c_spmv.split(var("i"), 512, j0, j1);
    b0.split(var("i"), 512, j0, j1);
    b1.split(var("i"), 512, j0, j1);
    sync_stage_2.split(j, 512, j0, j1);
    sdata_alloc.split(j, 512, j0, j1);
    sdata_init.split(j, 512, j0, j1);
    mul.split(j, 512, j0, j1);
    reduction256.split(j, 512, j0, j1);
    sync256.split(j, 512, j0, j1);
    reduction128.split(j, 512, j0, j1);
    sync128.split(j, 512, j0, j1);
    reduction64.split(j, 512, j0, j1);
    sync64.split(j, 512, j0, j1);
    reduction32.split(j, 512, j0, j1);
    reduction16.split(j, 512, j0, j1);
    reduction8.split(j, 512, j0, j1);
    reduction4.split(j, 512, j0, j1);
    reduction2.split(j, 512, j0, j1);
    reduction1.split(j, 512, j0, j1);


    sync_stage_1.after(w, j1);
    b0.after(sync_stage_1, j1);
    b1.after(b0, j1);
    c_spmv.after(b1, j1);
    sync_stage_2.after(c_spmv, j1);
    sdata_alloc.after(sync_stage_2, j1);
    sdata_init.after(sdata_alloc, j1);
    mul.after(sdata_init, j1);
    reduction256.after(mul, j1);
    sync256.after(reduction256, j1);
    reduction128.after(sync256, j1);
    sync128.after(reduction128, j1);
    reduction64.after(sync128, j1);
    sync64.after(reduction64, j1);
    reduction32.after(sync64, j1);
    reduction16.after(reduction32, j1);
    reduction8.after(reduction16, j1);
    reduction4.after(reduction8, j1);
    reduction2.after(reduction4, j1);
    reduction1.after(reduction2, j1);
    res_init.after(reduction1, computation::root);
    res.after(res_init, computation::root);

#else
    w.split(0, PARTITIONS);
    w.tag_parallel_level(0);
    w.split(1, B2);
    w.split(2, B1);
    w.tag_unroll_level(2);
    w.tag_vector_level(3, B1);

    // spmv schedule
    // ----------------------
    b0.split(0, PARTITIONS);
    b1.split(0, PARTITIONS);
    c_spmv.split(0, PARTITIONS);
    b0.split(1, B2);
    b1.split(1, B2);
    c_spmv.split(1, B2);
    c_spmv.tag_parallel_level(0);

 
    // dot schedule
    // ---------------------

    // Split (prepare for parallelization)
    mul.split(0, PARTITIONS);
    res.split(0, PARTITIONS);

    // Split (prepare for vectorization and split)
    mul.split(1, B2);
    res.split(1, B2);

    // Vectorization and unrolling
    mul.tag_vector_level(2, B2);
    res.tag_unroll_level(2);

    // parallelization
    res.tag_parallel_level(0);

    // Ordering
    res_init.after_low_level(w, -1);
    mul_alloc.after_low_level(res_init,1);
    b0.after_low_level(mul_alloc, 1);
    b1.after_low_level(b0, 2);
    c_spmv.after_low_level(b1,3);

    
    mul.after_low_level(c_spmv,1);
    res.after_low_level(mul, 1);
    res_global.after_low_level(res, -1);
#endif

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    // waxpby
    buffer b_SIZES("b_SIZES", {tiramisu::expr(1)}, p_int32, a_input, &cg);
    buffer b_x("b_x", {tiramisu::var("M")}, p_float64, a_input, &cg);
    buffer b_y("b_y", {tiramisu::var("M")}, p_float64, a_input, &cg);
    buffer b_alpha("b_alpha", {tiramisu::expr(1)}, p_float64, a_input, &cg);
    buffer b_beta("b_beta", {tiramisu::expr(1)}, p_float64, a_input, &cg);
    buffer b_w("b_w", {tiramisu::var("M")}, p_float64, a_output, &cg);

    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    alpha.set_access("{alpha[0]->b_alpha[0]}");
    beta.set_access("{beta[0]->b_beta[0]}");

#ifdef USE_GPU
    buffer b_w_gpu("b_w_gpu", {tiramisu::var("M")}, p_float64, a_temporary, &cg);
    buffer b_x_gpu("b_x_gpu", {tiramisu::var("M")}, p_float64, a_temporary, &cg);
    buffer b_y_gpu("b_y_gpu", {tiramisu::var("M")}, p_float64, a_temporary, &cg);

    b_w_gpu.tag_gpu_global();
    b_x_gpu.tag_gpu_global();
    b_y_gpu.tag_gpu_global();
    x.set_access("{x[j]->b_x_gpu[j]}");
    y.set_access("{y[j]->b_y_gpu[j]}");
    w.set_access("{w[j]->b_w_gpu[j]}");
#else
    x.set_access("{x[j]->b_x[j]}");
    y.set_access("{y[j]->b_y[j]}");
    w.set_access("{w[j]->b_w[j]}");
#endif

    // spmv
    buffer b_row_start("b_row_start", {tiramisu::expr(N)}, p_int32, a_input, &cg);
    buffer b_col_idx("b_col_idx", {tiramisu::expr(N)}, p_int32, a_input, &cg);
    buffer b_values("b_values", {tiramisu::expr((N*N))}, p_float64, a_input, &cg);
    buffer b_spmv("b_spmv", {tiramisu::expr(N*N)}, p_float64, a_output, &cg);

#ifdef USE_GPU
    buffer b_row_start_gpu("b_row_start_gpu", {tiramisu::expr(N)}, p_int32, a_temporary, &cg);
    buffer b_col_idx_gpu("b_col_idx_gpu", {tiramisu::expr(N)}, p_int32, a_temporary, &cg);
    buffer b_values_gpu("b_values_gpu", {tiramisu::expr((N*N))}, p_float64, a_temporary, &cg);
    buffer b_spmv_gpu("b_spmv_gpu", {tiramisu::expr(N*N)}, p_float64, a_temporary, &cg);
    b_row_start_gpu.tag_gpu_global();
    b_col_idx_gpu.tag_gpu_global();
    b_values_gpu.tag_gpu_global();
    b_spmv_gpu.tag_gpu_global();
    c_row_start.set_access("{c_row_start[i]->b_row_start_gpu[i]}");
    c_col_idx.set_access("{c_col_idx[j]->b_col_idx_gpu[j]}");
    c_values.set_access("{c_values[j]->b_values_gpu[j]}");
    c_spmv.set_access("{c_spmv[i,j]->b_spmv_gpu[i]}");
    c_spmv_wrapper.set_access("{c_spmv_wrapper[i]->b_spmv_gpu[i]}");
#else
    c_row_start.set_access("{c_row_start[i]->b_row_start[i]}");
    c_col_idx.set_access("{c_col_idx[j]->b_col_idx[j]}");
    c_values.set_access("{c_values[j]->b_values[j]}");
    c_spmv.set_access("{c_spmv[i,j]->b_spmv[i]}");
    c_spmv_wrapper.set_access("{c_spmv_wrapper[i]->b_spmv[i]}");
#endif

    // dot
    buffer b_res_global("b_res_global", {tiramisu::expr((int) 1)}, p_float64, a_output, &cg);

#ifdef USE_GPU
    buffer b_partial_res_gpu("b_partial_res_gpu", {(M_CST + 511) / 512}, p_float64, a_temporary, &cg);
    b_partial_res_gpu.tag_gpu_global();
    buffer b_partial_res("b_partial_res", {(M_CST + 511) / 512}, p_float64, a_temporary, &cg);
    //b_partial_res.set_auto_allocate(false);
    //computation allocate_partial{"{allocate_partial[0]}", allocate(b_partial_res), true, p_none, &cg};
    computation assign_partial{"{assign_partial[0]}", 0., true, p_float64, &cg};
    assign_partial.set_access("{assign_partial[i] -> b_partial_res[i]}");
    buffer sdata{"sdata", {512}, p_float64, a_temporary, &cg};
    sdata.tag_gpu_shared();
    sdata_init.set_access("{sdata_init[j] -> sdata[j % 512]}");
    mul.set_access("{mul[j] -> sdata[j % 512]}");
    reduction256.set_access("{reduction256[j] -> sdata[j % 512]}");
    reduction128.set_access("{reduction128[j] -> sdata[j % 512]}");
    reduction64.set_access("{reduction64[j] -> sdata[j % 512]}");
    reduction32.set_access("{reduction32[j] -> sdata[j % 512]}");
    reduction16.set_access("{reduction16[j] -> sdata[j % 512]}");
    reduction8.set_access("{reduction8[j] -> sdata[j % 512]}");
    reduction4.set_access("{reduction4[j] -> sdata[j % 512]}");
    reduction2.set_access("{reduction2[j] -> sdata[j % 512]}");
    reduction1.set_access("{reduction1[j] -> b_partial_res_gpu[j/512]}");
    reduced.set_access("{reduced[i] -> b_partial_res[i]}");
    res_init.set_access("{res_init[i] -> b_res_global[0]}");
    res.set_access("{res[i] -> b_res_global[0]}");
#else
    buffer b_mul("b_mul", {tiramisu::expr((int) B2)}, p_float64, a_temporary, &cg);
    b_mul.set_auto_allocate(false);
    buffer b_res("b_res", {tiramisu::var("M")/tiramisu::expr((int) PARTITIONS)}, p_float64, a_temporary, &cg);
    b_res.set_auto_allocate(false);
    mul.set_access("{mul[j]->b_mul[j%"+std::to_string(B2)+"]}");
    res_global.set_access("{res_global[j]->b_res_global[0]}");
    res_init.set_access("{res_init[t]->b_res[t]}");
    res.set_access("{res[j]->b_res[j/"+std::to_string(PARTITIONS)+"]}");
#endif

#ifdef USE_GPU
    computation copy_x{"{copy_x[0]}", memcpy(b_x, b_x_gpu), true, p_none, &cg};
    computation copy_y{"{copy_y[0]}", memcpy(b_y, b_y_gpu), true, p_none, &cg};
    computation copy_row_start{"{copy_row_start[0]}", memcpy(b_row_start, b_row_start_gpu), true, p_none, &cg};
    computation copy_col_idx{"{copy_col_idx[0]}", memcpy(b_col_idx, b_col_idx_gpu), true, p_none, &cg};
    computation copy_values{"{copy_values[0]}", memcpy(b_values, b_values_gpu), true, p_none, &cg};
    computation copy_spmv{"{copy_spmv[0]}", memcpy(b_spmv_gpu, b_spmv), true, p_none, &cg};
    computation copy_partial_res_gpu{"{copy_partial_res_gpu[0]}", memcpy(b_partial_res_gpu, b_partial_res), true, p_none, &cg};

    //copy_x.after(allocate_partial, computation::root);
    copy_x.after(assign_partial, computation::root);
    copy_y.after(copy_x, computation::root);
    copy_row_start.after(copy_y, computation::root);
    copy_col_idx.after(copy_row_start, computation::root);
    copy_values.after(copy_col_idx, computation::root);
    copy_values.before(w, computation::root);

    copy_partial_res_gpu.between(reduction1, computation::root, res_init, computation::root);
    copy_spmv.after(res, computation::root);
#endif


    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    cg.set_arguments({&b_SIZES, &b_alpha, &b_x, &b_beta, &b_y, &b_w, &b_row_start, &b_col_idx, &b_values, &b_spmv, &b_res_global});
    cg.gen_time_space_domain();
    cg.gen_isl_ast();
    cg.gen_c_code();
    cg.gen_cuda_stmt();
    cg.gen_halide_stmt();
    cg.gen_halide_obj("generated_cg.o");

    return 0;
}

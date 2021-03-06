#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"

/* CSR SpMV.
for (i = 0; i < M; i++)
    for (j = row_start[i]; j<row_start[i+1]; j++)
    {
        y[i] += values[j] * x[col_idx[j]];
    }
*/

/* CSR SpMV Simplified.
for (i = 0; i < M; i++)
    int b0 = row_start[i];
    int b1 = row_start[i+1];

    for (j = b0; j < b1; j++)
    {
        int t = col_idx[j];
        y[i] += values[j] * x[t];
    }
*/

#define SIZE0 1000

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int32);


    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------


    function spmv("spmv");

    expr e_M = expr((int32_t) SIZE0);
    constant M("M", e_M, p_int32, true, NULL, 0, &spmv);

    computation c_row_start("[M]->{c_row_start[i]: 0<=i<M}", expr(), false, p_uint8, &spmv);
    computation c_col_idx("[b0,b1]->{c_col_idx[j]: b0<=j<b1}", expr(), false, p_uint8, &spmv);
    computation c_values("[b0,b1]->{c_values[j]: b0<=j<b1}", expr(), false, p_uint8, &spmv);
    computation c_x("[M,b0,b1]->{c_x[j]: b0<=j<b1}", expr(), false, p_uint8, &spmv);

    computation c_y("[M,b0,b1]->{c_y[i,j]: 0<=i<M and b0<=j<b1}", expr(), true, p_uint8, &spmv);

    spmv.set_context_set("[M,b0,b1]->{: M>0 and b0>0 and b1>0 and b1>b0 and b1%4=0}");

    expr e_t = c_col_idx(var("j"));
    constant t("t", e_t, p_int32, false, &c_y, 1, &spmv);
    expr e_b1 = c_row_start(var("i") + 1);
    constant b1("b1", e_b1, p_int32, false, &t, 0, &spmv);
    expr e_b0 = c_row_start(var("i"));
    constant b0("b0", e_b0, p_int32, false, &b1, 0, &spmv);

    expr e_y = c_y(var("i")) + c_values(var("j")) * c_x(var("t"));
    c_y.set_expression(e_y);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


    b0.set_low_level_schedule("[M]->{b0[i]->b0[0,0,i,0,0,0]: 0<=i<M}");
    b1.set_low_level_schedule("[M]->{b1[i]->b1[0,0,i,1,0,0]: 0<=i<M}");
    t.set_low_level_schedule("[M,b0,b1]->{t[i,j]->t[0,0,i,2,j1,1,j2,0]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and b0<=j<(b1/4) and b1%4=0 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1;   t[i,j]->t[0,0,i,2,j1,0,j2,0]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and (b1/4)<=j<b1 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1;}");
    c_y.set_low_level_schedule("[M,b0,b1]->{c_y[i,j]->c_y[0,0,i,2,j1,1,j2,1]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and b0<=j<(b1/4) and b1%4=0 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1; c_y[i,j]->c_y[0,0,i,2,j1,0,j2,1]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and (b1/4)<=j<b1 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1;}");
    c_y.tag_parallel_level(var("i"));


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------


    buffer b_row_start("b_row_start", {tiramisu::expr(SIZE0)}, p_uint8, a_input, &spmv);
    buffer b_col_idx("b_col_idx", {tiramisu::expr(SIZE0)}, p_uint8, a_input, &spmv);
    buffer b_values("b_values", {tiramisu::expr(SIZE0 * SIZE0)}, p_uint8, a_input, &spmv);
    buffer b_x("b_x", {tiramisu::expr(SIZE0 * SIZE0)}, p_uint8, a_input, &spmv);
    buffer b_y("b_y", {tiramisu::expr(SIZE0 * SIZE0)}, p_uint8, a_output, &spmv);

    c_row_start.set_access("{c_row_start[i]->b_row_start[i]}");
    c_col_idx.set_access("{c_col_idx[j]->b_col_idx[j]}");
    c_values.set_access("{c_values[j]->b_values[j]}");
    c_x.set_access("{c_x[j]->b_x[j]}");
    c_y.set_access("{c_y[i,j]->b_y[i]}");


    // -------------------------------------------------------
    // Code Generator
    // -------------------------------------------------------


    spmv.set_arguments({&b_row_start, &b_col_idx, &b_values, &b_x, &b_y});
    spmv.gen_time_space_domain();
    spmv.gen_isl_ast();
    spmv.gen_halide_stmt();
    spmv.gen_halide_obj("build/generated_fct_tutorial_04.o");

    // Some debugging
    spmv.dump_halide_stmt();

    return 0;
}

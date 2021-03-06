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

/* Sequence of computations.

for (i = 0; i < M; i++)
  S0(i) = 4;
  S1(i) = 3;
  for (j = 0; j < N; j++)
    S2(i, j) = 2;
  S3(i) = 1;
*/

#define SIZE0 10

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int32);


    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------


    function sequence("sequence");
    expr e_M = expr((int32_t) SIZE0);
    constant M("M", e_M, p_int32, true, NULL, 0, &sequence);
    computation c0("[M]->{c0[i]: 0<=i<M}", tiramisu::expr((uint8_t) 4), true, p_uint8, &sequence);
    computation c1("[M]->{c1[i]: 0<=i<M}", tiramisu::expr((uint8_t) 3), true, p_uint8, &sequence);
    computation c2("[M]->{c2[i,j]: 0<=i<M and 0<=j<M}", tiramisu::expr((uint8_t) 2), true, p_uint8,
                   &sequence);
    computation c3("[M]->{c3[i]: 0<=i<M}", tiramisu::expr((uint8_t) 1), true, p_uint8, &sequence);

    sequence.dump_iteration_domain();

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    var i("i");
    c1.after(c0, i);
    c2.after(c1, i);
    c3.after(c2, i);


    sequence.dump_schedule();

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------


    buffer b0("b0", {tiramisu::expr(SIZE0)}, p_uint8, a_output, &sequence);
    buffer b1("b1", {tiramisu::expr(SIZE0)}, p_uint8, a_output, &sequence);
    buffer b2("b2", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE0)}, p_uint8, a_output,
              &sequence);
    buffer b3("b3", {tiramisu::expr(SIZE0)}, p_uint8, a_output, &sequence);

    c0.set_access("{c0[i]->b0[i]}");
    c1.set_access("{c1[i]->b1[i]}");
    c2.set_access("{c2[i,j]->b2[i,j]}");
    c3.set_access("{c3[i]->b3[i]}");


    // -------------------------------------------------------
    // Code Generator
    // -------------------------------------------------------


    sequence.set_arguments({&b0, &b1, &b2, &b3});
    sequence.gen_time_space_domain();
    sequence.dump_trimmed_time_processor_domain();
    sequence.gen_isl_ast();
    sequence.gen_halide_stmt();
    sequence.gen_halide_obj("build/generated_fct_tutorial_05.o");

    // Some debugging
    sequence.dump(true);
    sequence.dump_halide_stmt();

    return 0;
}

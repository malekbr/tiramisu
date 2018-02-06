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

#include "wrapper_test_94.h"

using namespace tiramisu;

/**
 * TODO: describe test
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(p_int64);

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    // TODO: define computations
    tiramisu::constant N("N", 40L, tiramisu::global::get_loop_iterator_default_data_type(), true, nullptr, 0,  &function0);
    tiramisu::constant M("M", 50L, tiramisu::global::get_loop_iterator_default_data_type(), true, nullptr, 0,  &function0);
    var i("i"), j("j");
    tiramisu::computation S0("[N, M] -> {S0[i, j] : 0 <= i < N and 0 <= j < M}", expr(), false, p_float64, &function0);
    tiramisu::computation S1("[N, M] -> {S1[i, j] : 0 <= i < N and 0 <= j < M}", expr(), false, p_float64, &function0);

    tiramisu::computation S2("[N, M] -> {S2[i, j] : 0 <= i < N and 0 <= j < M}", S0(i, j) + S1(i, j), true, p_float64, &function0);
    tiramisu::computation S3i("[N, M] -> {S3i[i, j] : 0 <= i < N and 0 <= j < M}", -S2(i, j), true, p_float64, &function0);
    tiramisu::computation S3u("[N, M] -> {S3u[i, j, k] : 0 <= i < N and 0 <= j < M and 0 <= k < M}", S3i(i, j) + S2(i, j), true, p_float64, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // TODO: add layer II if necessary
    S3i.after(S2, j);
    S3u.after(S3i, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // TODO: modify layer III if necessary
    tiramisu::buffer B0("B0", {N, M}, p_float64, a_input, &function0);
    tiramisu::buffer B1("B1", {N, M}, p_float64, a_input, &function0);
    tiramisu::buffer B2("B2", {N, M, 1}, p_float64, a_temporary, &function0);
    tiramisu::buffer B3("B3", {N, M}, p_float64, a_output, &function0);

    S0.set_access("{S0[i, j] -> B0[i, j]}");
    S1.set_access("{S1[i, j] -> B1[i, j]}");
    S2.set_access("{S2[i, j] -> B2[i, j, 0]}");
    S3i.set_access("{S3i[i, j] -> B3[i, j]}");
    S3u.set_access("{S3u[i, j, k] -> B3[i, j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&B0, &B1, &B3});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_cuda_stmt();
    // function0.gen_halide_stmt();
    // function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}

#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>

#include <tiramisu/core.h>
#include <tiramisu/debug.h>

#include "benchmarks.h"
#include <Halide.h>
#include <string.h>

/* dot product between two vectors.

inputs:
--------
- y[]
- x[]

res[0] = 0;
for (i = 0; i < M; i++)
        res[0] = res[0] + y[i] * x[i];
*/

using namespace tiramisu;

constexpr auto bs = 512;
#define BLOCK_SIZE "512"

int main(int argc, char **argv) {
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function dot("dot_gpu");

    // Inputs
    computation sizes("[M]->{sizes[0]}", tiramisu::expr(), false, p_int32, &dot);
    constant M_CST("M", sizes(0), p_int32, true, NULL, 0, &dot);
    computation x("[M]->{x[j]}", tiramisu::expr(), false, p_float64, &dot);
    computation y("[M]->{y[j]}", tiramisu::expr(), false, p_float64, &dot);
    constant MR{"MR", expr{o_max, (M_CST + bs - 1) / bs, 1}, p_int32, true, nullptr, 0, &dot};
    constant MR2{"MR2", expr{o_max, (MR + bs - 1) / bs, 1}, p_int32, true, nullptr, 0, &dot};
    computation reduced_1{"[MR] -> {reduced_1[i]: 0 <= i < MR}", expr(), false, p_float64, &dot};
    computation reduced_2{"[MR2] -> {reduced_2[i]: 0 <= i < MR2}", expr(), false, p_float64, &dot};

    tiramisu::var i("i"), i0("i0"), i1("i1"), j("j"), j0("j0"), j1("j1");

#define GPU_LIM "ceil(M / " BLOCK_SIZE ") * " BLOCK_SIZE

    buffer dot_gpu{"dot_gpu", {bs}, p_float64, a_temporary, &dot};
    dot_gpu.tag_gpu_shared();

    computation shared_dot_dec{"[M] -> {shared_dot_dec[i]: 0 <= i < " GPU_LIM "}", allocate(dot_gpu), true, p_float64, &dot};
    shared_dot_dec.split(i, bs, i0, i1);
    computation shared_dot{"[M] -> {shared_dot[i]: 0 <= i < " GPU_LIM "}", expr{o_select, i < M_CST, x(i) * y(i), 0.}, true, p_float64, &dot};

    shared_dot.set_access("{shared_dot[i] -> dot_gpu[i % " BLOCK_SIZE "]}");

    shared_dot.split(i, bs, i0, i1);

    shared_dot.after(shared_dot_dec, i1);

    std::vector<computation *> perform_add_dot;
    computation *last_sync = &shared_dot;

    for (int s = bs / 2; s > 0; s >>= 1) {
        auto *comp =
            new computation{"[M] -> {add_dot_" + std::to_string(s) + "[i] : 0 <= i < " GPU_LIM " and i % " BLOCK_SIZE " < " + std::to_string(s) + "}",
                            shared_dot(i) + shared_dot(i + s), true, p_float64, &dot};
        comp->split(i, bs, i0, i1);
        auto *sync_comp =
            new computation{"[M] -> {sync_dot_" + std::to_string(s) + "[i] : 0 <= i < " GPU_LIM "}", tiramisu::sync{}, true, p_none, &dot};
        sync_comp->split(i, bs, i0, i1);

        comp->after(*last_sync, i1);
        sync_comp->after(*comp, i1);
        last_sync = sync_comp;

        perform_add_dot.push_back(comp);

        comp->set_access("{" + comp->get_name() + "[i] -> dot_gpu[i % " BLOCK_SIZE "]}");
    }

    computation assign_back{"[M] -> {assign_back[i]: 0 <= i <  " GPU_LIM " and i % " BLOCK_SIZE " = 0}", shared_dot(i), true, p_float64, &dot};
    assign_back.split(i, bs, i0, i1);

    assign_back.after(*last_sync, i1);

#define GPU_LIM "ceil(MR / " BLOCK_SIZE ") * " BLOCK_SIZE

    buffer sum_gpu{"sum_gpu", {bs}, p_float64, a_temporary, &dot};
    sum_gpu.tag_gpu_shared();

    computation shared_sum_dec{"[MR] -> {shared_sum_dec[i]: 0 <= i < " GPU_LIM "}", allocate(sum_gpu), true, p_float64, &dot};
    shared_sum_dec.split(i, bs, i0, i1);

    computation shared_sum{"[MR] -> {shared_sum[i]: 0 <= i < " GPU_LIM "}", expr{o_select, i < MR, reduced_1(i), 0.}, true, p_float64, &dot};
    shared_sum.set_access("{shared_sum[i] -> sum_gpu[i % " BLOCK_SIZE "]}");
    shared_sum.split(i, bs, i0, i1);
    shared_sum.after(shared_sum_dec, i1);

    std::vector<computation *> perform_add_sum;
    last_sync = &shared_sum;

    for (int s = bs / 2; s > 0; s >>= 1) {
        auto *comp = new computation{"[MR] -> {add_sum_" + std::to_string(s) + "[i] : 0 <= i < " GPU_LIM " and i % " BLOCK_SIZE " < " +
                                         std::to_string(s) + "}",
                                     shared_sum(i) + shared_sum(i + s), true, p_float64, &dot};
        comp->split(i, bs, i0, i1);
        auto *sync_comp =
            new computation{"[MR] -> {sync_comp_" + std::to_string(s) + "[i] : 0 <= i < " GPU_LIM "}", tiramisu::sync{}, true, p_none, &dot};
        sync_comp->split(i, bs, i0, i1);

        comp->after(*last_sync, i1);
        sync_comp->after(*comp, i1);
        last_sync = sync_comp;

        perform_add_sum.push_back(comp);
        comp->set_access("{" + comp->get_name() + "[i] -> sum_gpu[i % " BLOCK_SIZE "]}");
    }

    computation assign_sum{"[MR] -> {assign_sum[i]: 0 <= i < " GPU_LIM " and i % " BLOCK_SIZE " = 0}", shared_sum(i), true, p_float64, &dot};
    assign_sum.split(i, bs, i0, i1);

    assign_sum.after(*last_sync, i1);

    computation init_final_sum{"{init_final_sum[0]}", 0., true, p_float64, &dot};
    computation reduce_result{"[MR2] -> {reduce_result[i]: 0 <= i < MR2}", init_final_sum(0) + reduced_2(i), true, p_float64, &dot};

    // -----------------------------------------------------------------
    // Layer II
    // -----------------------------------------------------------------

    shared_sum_dec.after(assign_back, computation::root);
    shared_sum_dec.tag_gpu_level(i0, i1);

    shared_dot_dec.tag_gpu_level(i0, i1);

    init_final_sum.after(assign_sum, computation::root);
    reduce_result.after(init_final_sum, computation::root);

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    buffer sizes_host{"sizes_host", {1}, p_int32, a_input, &dot};
    sizes.set_access("{sizes[i] -> sizes_host[i]}");
    buffer x_host{"x_host", {M_CST}, p_float64, a_input, &dot};
    buffer y_host{"y_host", {M_CST}, p_float64, a_input, &dot};
    buffer x_gpu{"x_gpu", {M_CST}, p_float64, a_temporary, &dot};
    buffer y_gpu{"y_gpu", {M_CST}, p_float64, a_temporary, &dot};
    x_gpu.tag_gpu_global();
    y_gpu.tag_gpu_global();
    x.set_access("{x[i] -> x_gpu[i]}");
    y.set_access("{y[i] -> y_gpu[i]}");

    buffer reduced_1_gpu{"reduced_1_gpu", {MR}, p_float64, a_temporary, &dot};
    reduced_1_gpu.tag_gpu_global();
    reduced_1.set_access("{reduced_1[i] -> reduced_1_gpu[i]}");
    assign_back.set_access("{assign_back[i] -> reduced_1_gpu[i / " BLOCK_SIZE "]}");

    buffer reduced_2_gpu{"reduced_2_gpu", {MR2}, p_float64, a_temporary, &dot};
    buffer reduced_2_host{"reduced_2_host", {MR2}, p_float64, a_temporary, &dot};
    reduced_2_gpu.tag_gpu_global();
    reduced_2.set_access("{reduced_2[i] -> reduced_2_host[i]}");
    assign_sum.set_access("{assign_sum[i] -> reduced_2_gpu[i / " BLOCK_SIZE "]}");

    buffer result_host{"result_host", {1}, p_float64, a_output, &dot};
    init_final_sum.set_access("{init_final_sum[i] -> result_host[0]}");
    reduce_result.set_access("{reduce_result[i] -> result_host[0]}");


    computation copy_x{"{copy_x[0]}", memcpy(x_host, x_gpu), true, p_none, &dot};
    computation copy_y{"{copy_y[0]}", memcpy(y_host, y_gpu), true, p_none, &dot};
    computation copy_reduced{"{copy_reduced[0]}", memcpy(reduced_2_gpu, reduced_2_host), true, p_none, &dot};

    copy_x.before(copy_y, computation::root);
    copy_y.before(shared_dot_dec, computation::root);

    copy_reduced.between(assign_sum, computation::root, init_final_sum, computation::root);

    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    dot.set_arguments({&sizes_host, &x_host, &y_host, &result_host});
    dot.gen_time_space_domain();
    dot.gen_isl_ast();
    dot.gen_c_code();
    dot.gen_cuda_stmt();
    dot.gen_halide_stmt();
    dot.gen_halide_obj("generated_dot_gpu.o");

    return 0;
}

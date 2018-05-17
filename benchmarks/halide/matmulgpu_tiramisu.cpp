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


using namespace tiramisu;

constexpr auto it_type = p_int32;
constexpr auto data_type = p_float32;
constexpr auto block_size = 16;
#define BLOCK_SIZE "16"

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function matmulgpu_tiramisu("matmulgpu_tiramisu");

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1"), k("k"), k0("k0"), k1("k1");

    // Layer I

    computation sizes{"{sizes[i]: 0 <= i < 3}", expr(), false, it_type, &matmulgpu_tiramisu};
    constant N{"N", sizes(0), it_type, true, nullptr, 0, &matmulgpu_tiramisu};
    constant D{"D", sizes(1), it_type, true, nullptr, 0, &matmulgpu_tiramisu};
    constant M{"M", sizes(2), it_type, true, nullptr, 0, &matmulgpu_tiramisu};

    computation A{"[N, D] -> {A[i, j]: 0 <= i < N and 0 <= j < D}",
                   expr(), false, data_type, &matmulgpu_tiramisu};
    computation B{"[D, M] -> {B[i, j]: 0 <= i < D and 0 <= j < M}",
                   expr(), false, data_type, &matmulgpu_tiramisu};
    computation init{"[N, M] -> {init[i, j]: 0 <= i < N and 0 <= j < M}", 0., true, data_type, &matmulgpu_tiramisu};
    computation multiply{"[N, M, D] -> {multiply[i, j, k]: 0 <= i < N and 0 <= j < M and 0 <= k < D}", 0., true, data_type, &matmulgpu_tiramisu};
    multiply.set_expression(multiply(i, j, k) + A(i, k) * B(k, j));

    // Layer II
    init.interchange(j, i);
    multiply.interchange(j, i);
    init.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    multiply.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    multiply.split(k, block_size, k0, k1);

//    multiply.after(init, j1);

    // Layer III
    buffer sizes_host{"sizes_host", {3}, it_type, a_input, &matmulgpu_tiramisu};
    sizes.set_access("{sizes[i] -> sizes_host[i]}");
    buffer sum_gpu{"sum_gpu", {1}, data_type, a_temporary, &matmulgpu_tiramisu};
    sum_gpu.tag_gpu_register();
    init.set_access("{init[i, j] -> sum_gpu[0]}");
    multiply.set_access("{multiply[i, j, k] -> sum_gpu[0]}");
    buffer A_host{"A_host", {N, D}, data_type, a_input, &matmulgpu_tiramisu};
    buffer B_host{"B_host", {D, M}, data_type, a_input, &matmulgpu_tiramisu};
    buffer C_host{"C_host", {N, M}, data_type, a_output, &matmulgpu_tiramisu};
    buffer A_gpu{"A_gpu", {N, D}, data_type, a_temporary, &matmulgpu_tiramisu};
    buffer B_gpu{"B_gpu", {D, M}, data_type, a_temporary, &matmulgpu_tiramisu};
    buffer C_gpu{"C_gpu", {N, M}, data_type, a_temporary, &matmulgpu_tiramisu};
    A_gpu.tag_gpu_global();
    B_gpu.tag_gpu_global();
    C_gpu.tag_gpu_global();
    buffer A_shared{"A_shared", {block_size, block_size}, data_type, a_temporary, &matmulgpu_tiramisu};
    buffer B_shared{"B_shared", {block_size, block_size}, data_type, a_temporary, &matmulgpu_tiramisu};
    A_shared.tag_gpu_shared();
    B_shared.tag_gpu_shared();

    A.set_access("{A[i, j] -> A_shared[i % " BLOCK_SIZE ", j % " BLOCK_SIZE "]}");
    B.set_access("{B[i, j] -> B_shared[i % " BLOCK_SIZE ", j % " BLOCK_SIZE "]}");


    // Layer IV
    computation A_global{"[N, D] -> {A_global[i, j]: 0 <= i < N and 0 <= j < D}",
                   expr(), false, data_type, &matmulgpu_tiramisu};
    computation B_global{"[M, D] -> {B_global[i, j]: 0 <= i < D and 0 <= j < M}",
                   expr(), false, data_type, &matmulgpu_tiramisu};
    A_global.set_access("{A_global[i, j] -> A_gpu[i, j]}");
    B_global.set_access("{B_global[i, j] -> B_gpu[i, j]}");
    computation shared_a{"[N, M, D] -> {shared_a[j, i, k]: 0 <= i < N and 0 <= j < M and 0 <= k < D and k % " BLOCK_SIZE " = j % " BLOCK_SIZE "}", A_global(i, k), true, data_type, &matmulgpu_tiramisu};
    computation shared_b{"[N, M, D] -> {shared_b[j, i, k]: 0 <= i < N and 0 <= j < M and 0 <= k < D and k % " BLOCK_SIZE " = i % " BLOCK_SIZE "}", B_global(k, j), true, data_type, &matmulgpu_tiramisu};
    computation declare_sum{"[N, M] -> {declare_sum[j, i]: 0 <= i < ceil(N/" BLOCK_SIZE ")*" BLOCK_SIZE " and 0 <= j < ceil(M/" BLOCK_SIZE ")*" BLOCK_SIZE "}", allocate(sum_gpu), true, data_type, &matmulgpu_tiramisu};
    computation declare_a_shared{"[N, M] -> {declare_a_shared[j, i]: 0 <= i < ceil(N/" BLOCK_SIZE ")*" BLOCK_SIZE " and 0 <= j < ceil(M/" BLOCK_SIZE ")*" BLOCK_SIZE "}", allocate(A_shared), true, data_type, &matmulgpu_tiramisu};
    computation declare_b_shared{"[N, M] -> {declare_b_shared[j, i]: 0 <= i < ceil(N/" BLOCK_SIZE ")*" BLOCK_SIZE " and 0 <= j < ceil(M/" BLOCK_SIZE ")*" BLOCK_SIZE "}", allocate(B_shared), true, data_type, &matmulgpu_tiramisu};
    computation sync1{"[N, M, D] -> {sync1[j, i, k]: 0 <= i < ceil(N/" BLOCK_SIZE ")*" BLOCK_SIZE " and 0 <= j < ceil(M/" BLOCK_SIZE ")*" BLOCK_SIZE " and 0 <= k < D and k % " BLOCK_SIZE " = 0}", tiramisu::sync{}, true, data_type, &matmulgpu_tiramisu};
    computation sync2{"[N, M, D] -> {sync2[j, i, k]: 0 <= i < ceil(N/" BLOCK_SIZE ")*" BLOCK_SIZE " and 0 <= j < ceil(M/" BLOCK_SIZE ")*" BLOCK_SIZE " and 0 <= k < D and k % " BLOCK_SIZE " = 0}", tiramisu::sync{}, true, data_type, &matmulgpu_tiramisu};
    computation reassign{"[N, M, D] -> {reassign[j, i]: 0 <= i < N and 0 <= j < M}", init(i, j), true, data_type, &matmulgpu_tiramisu};
    reassign.set_access("{reassign[j, i] -> C_gpu[i, j]}");

    shared_a.set_access("{shared_a[j, i, k] -> A_shared[i % " BLOCK_SIZE ", k % " BLOCK_SIZE "]}");
    shared_b.set_access("{shared_b[j, i, k] -> B_shared[k % " BLOCK_SIZE ", j % " BLOCK_SIZE "]}");

    shared_a.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    shared_b.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    sync1.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    sync2.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    declare_sum.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    declare_a_shared.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    declare_b_shared.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    reassign.tile(j, i, block_size, block_size, i0, j0, i1, j1);
    shared_a.split(k, block_size, k0, k1);
    shared_b.split(k, block_size, k0, k1);
    sync1.split(k, block_size, k0, k1);
    sync2.split(k, block_size, k0, k1);

    declare_a_shared.after(declare_sum, j1);
    declare_b_shared.after(declare_a_shared, j1);
    init.after(declare_b_shared, j1);
    shared_a.after(init, j1);
    shared_b.after(shared_a, k0);
    sync1.after(shared_b, k0);
    multiply.after(sync1, k0);
    sync2.after(multiply, k0);
    reassign.after(sync2, j1);

    declare_sum.tag_gpu_level(i0, j0, i1, j1);

    computation copy_a{"{copy_a[0]}", memcpy(A_host, A_gpu), true, p_none, &matmulgpu_tiramisu};
    computation copy_b{"{copy_b[0]}", memcpy(B_host, B_gpu), true, p_none, &matmulgpu_tiramisu};
    computation copy_c{"{copy_c[0]}", memcpy(C_gpu, C_host), true, p_none, &matmulgpu_tiramisu};

    copy_a.before(copy_b, computation::root);
    copy_b.before(declare_sum, computation::root);
    copy_c.after(reassign, computation::root);



    matmulgpu_tiramisu.set_arguments({&sizes_host, &A_host, &B_host, &C_host});
    matmulgpu_tiramisu.gen_time_space_domain();
    matmulgpu_tiramisu.gen_isl_ast();
    matmulgpu_tiramisu.gen_c_code();
    matmulgpu_tiramisu.gen_cuda_stmt();
    matmulgpu_tiramisu.gen_halide_stmt();
    matmulgpu_tiramisu.dump_halide_stmt();
    matmulgpu_tiramisu.gen_halide_obj("build/generated_fct_matmulgpu.o");

    return 0;
}


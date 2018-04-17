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

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int32);

    tiramisu::function rgbyuv420("recfiltergpu");

    buffer input_host{"input_host", {45}, p_float64, a_input, &rgbyuv420};
    buffer input_gpu{"input_gpu", {45}, p_float64, a_temporary, &rgbyuv420};
    input_gpu.tag_gpu_global();

    computation copy{"{copy[i]: 0 <= i < 10}", memcpy(input_host, input_gpu), true, p_none, &rgbyuv420};


    // Add schedules.
    rgbyuv420.set_arguments({&input_host});
    rgbyuv420.gen_time_space_domain();
    rgbyuv420.gen_isl_ast();
    rgbyuv420.gen_cuda_stmt();
    rgbyuv420.gen_halide_stmt();
    rgbyuv420.dump_halide_stmt();
    rgbyuv420.gen_halide_obj("build/generated_fct_recfiltergpu.o");

    return 0;
}

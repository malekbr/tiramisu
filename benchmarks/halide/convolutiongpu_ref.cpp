#include "Halide.h"
#include "wrapper_convolutiongpu.h"

using namespace Halide;

int main(int argc, char* argv[]) {
    ImageParam in_b(UInt(8), 3, "input");
    ImageParam kernel(Float(32), 2, "kernel");

    Func convolutiongpu("convolutiongpu"), in;
    Var x("x"), y("y"), c("c"), x0, x1, y0, y1;

    in(x, y, c) = in_b(x, y, c);

    Expr e = 0.0f;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            e += cast<float>(in(x + i, y + j, c)) * kernel(i, j);
        }
    }

    convolutiongpu(x, y, c) = cast<uint8_t>(e);

    convolutiongpu.reorder(c, x, y);

    convolutiongpu.gpu_tile(x, y, x0, y0, x1, y1, 14, 14);

    in.reorder(c, x, y);
    in.compute_at(convolutiongpu, x0);
    //in.tile(x, y, x1, y1, 16, 16);
    in.gpu_threads(x, y);



    Halide::Target target = Halide::get_host_target();
    target.set_feature(Target::Feature::CUDA, true);

    convolutiongpu.compile_to_object("build/generated_fct_convolutiongpu_ref.o", {in_b, kernel}, "convolutiongpu_ref", target);

    convolutiongpu.compile_to_lowered_stmt("build/generated_fct_convolutiongpu_ref.txt", {in_b, kernel}, Text, target);

    return 0;
}

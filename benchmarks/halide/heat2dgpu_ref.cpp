#include "Halide.h"
using namespace Halide;

constexpr auto block_size = 16;

int main(int argc, char **argv) {
    ImageParam in_b(Float(32), 2, "input");

    float alpha = 0.3;
    float beta = 0.4;

    Func heat2d("heat2dgpu"), in;
    Var x("x"), y("y"), x0, y0, x1, y1;
    RVar rx0, ry0, rx1, ry1;

    in(x, y) = in_b(x, y);

    RDom    up(0,                 in_b.width(),   0,                   1            );
    RDom  down(0,                 in_b.width(),   in_b.height() - 1,   1            );
    RDom  left(0,                 1,              0,                   in_b.height());
    RDom right(in_b.width() -1,   1,              0,                   in_b.height());

    RDom r(1, in_b.width()-2, 1, in_b.height()-2);
    heat2d(up.x, up.y) = 0.0f;
    heat2d(down.x, down.y) = 0.0f;
    heat2d(left.x, left.y) = 0.0f;
    heat2d(right.x, right.y) = 0.0f;
    heat2d(r.x, r.y) = alpha * in(r.x, r.y) +
                       beta * (in(r.x+1, r.y) + in(r.x-1, r.y) + in(r.x, r.y+1) + in(r.x, r.y-1));

    heat2d.gpu_tile(up.x, x0, x1, block_size);
    heat2d.update(1).gpu_tile(down.x, x0, x1, block_size);
    heat2d.update(2).gpu_tile(left.y, x0, x1, block_size);
    heat2d.update(3).gpu_tile(right.y, x0, x1, block_size);
    heat2d.update(4).gpu_tile(r.x, r.y, rx0, ry0, rx1, ry1, block_size, block_size);

    std::cout << heat2d.update().dump_argument_list() << std::endl;
    in.compute_at(heat2d, ry1);

    Halide::Target target = Halide::get_host_target();
    target.set_feature(Target::Feature::CUDA, true);

    heat2d.compile_to_object("build/generated_fct_heat2dgpu_ref.o",
                             {in_b},
                             "heat2dgpu_ref",
                             target);

    heat2d.compile_to_lowered_stmt("build/generated_fct_heat2dgpu_ref.txt",
                                   {in_b},
                                   Text,
                                   target);
    return 0;
}

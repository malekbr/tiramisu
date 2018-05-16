#include "wrapper_heat2dgpu.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<float> input(Halide::Float(32), 10000, 20000);

    Halide::Buffer<int32_t> size(2);
    size(0) = input.extent(0);
    size(1) = input.extent(1);
    // Init randomly
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            input(x, y) = 1;
        }
    }

    Halide::Buffer<float> output1(input.width(), input.height());
    Halide::Buffer<float> output2(input.width(), input.height());

    // Warm up code.
    heat2dgpu_tiramisu(size.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
//    heat2dgpu_ref(input.raw_buffer(), output2.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        heat2dgpu_tiramisu(size.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    std::cout << median(duration_vector_1) << std::endl;

//    // Reference
//    for (int i=0; i<NB_TESTS; i++)
//    {
//        auto start2 = std::chrono::high_resolution_clock::now();
//        heat2dgpu_ref(input.raw_buffer(), output2.raw_buffer());
//        auto end2 = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
//        duration_vector_2.push_back(duration2);
//    }
//
//    print_time("performance_CPU.csv", "heat2dgpu",
//               {"Tiramisu", "Halide"},
//               {175.4, median(duration_vector_2)});

//    if (CHECK_CORRECTNESS)
//	compare_buffers_approximately("benchmark_heat2dgpu", output1, output2);


    return 0;
}

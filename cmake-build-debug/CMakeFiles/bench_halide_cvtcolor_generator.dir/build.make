# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2017.2.2/bin/cmake/bin/cmake

# The command to remove a file.
RM = /opt/clion-2017.2.2/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/malek/tiramisu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/malek/tiramisu/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/bench_halide_cvtcolor_generator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bench_halide_cvtcolor_generator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bench_halide_cvtcolor_generator.dir/flags.make

CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o: CMakeFiles/bench_halide_cvtcolor_generator.dir/flags.make
CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o: ../benchmarks/halide/cvtcolor_ref.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o -c /home/malek/tiramisu/benchmarks/halide/cvtcolor_ref.cpp

CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/malek/tiramisu/benchmarks/halide/cvtcolor_ref.cpp > CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.i

CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/malek/tiramisu/benchmarks/halide/cvtcolor_ref.cpp -o CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.s

CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.requires:

.PHONY : CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.requires

CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.provides: CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.requires
	$(MAKE) -f CMakeFiles/bench_halide_cvtcolor_generator.dir/build.make CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.provides.build
.PHONY : CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.provides

CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.provides.build: CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o


# Object files for target bench_halide_cvtcolor_generator
bench_halide_cvtcolor_generator_OBJECTS = \
"CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o"

# External object files for target bench_halide_cvtcolor_generator
bench_halide_cvtcolor_generator_EXTERNAL_OBJECTS =

../build/bench_halide_cvtcolor_generator: CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o
../build/bench_halide_cvtcolor_generator: CMakeFiles/bench_halide_cvtcolor_generator.dir/build.make
../build/bench_halide_cvtcolor_generator: ../Halide/lib/libHalide.a
../build/bench_halide_cvtcolor_generator: CMakeFiles/bench_halide_cvtcolor_generator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../build/bench_halide_cvtcolor_generator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_halide_cvtcolor_generator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bench_halide_cvtcolor_generator.dir/build: ../build/bench_halide_cvtcolor_generator

.PHONY : CMakeFiles/bench_halide_cvtcolor_generator.dir/build

CMakeFiles/bench_halide_cvtcolor_generator.dir/requires: CMakeFiles/bench_halide_cvtcolor_generator.dir/benchmarks/halide/cvtcolor_ref.cpp.o.requires

.PHONY : CMakeFiles/bench_halide_cvtcolor_generator.dir/requires

CMakeFiles/bench_halide_cvtcolor_generator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bench_halide_cvtcolor_generator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bench_halide_cvtcolor_generator.dir/clean

CMakeFiles/bench_halide_cvtcolor_generator.dir/depend:
	cd /home/malek/tiramisu/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/malek/tiramisu /home/malek/tiramisu /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug/CMakeFiles/bench_halide_cvtcolor_generator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bench_halide_cvtcolor_generator.dir/depend


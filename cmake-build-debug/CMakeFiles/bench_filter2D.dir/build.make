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
include CMakeFiles/bench_filter2D.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bench_filter2D.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bench_filter2D.dir/flags.make

../build/generated_fct_filter2D.o:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../build/generated_fct_filter2D.o"
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/bench_tiramisu_filter2D_generator

../build/generated_fct_filter2D_ref.o:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating ../build/generated_fct_filter2D_ref.o"
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/bench_halide_filter2D_generator

CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o: CMakeFiles/bench_filter2D.dir/flags.make
CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o: ../benchmarks/halide/wrapper_filter2D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o -c /home/malek/tiramisu/benchmarks/halide/wrapper_filter2D.cpp

CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/malek/tiramisu/benchmarks/halide/wrapper_filter2D.cpp > CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.i

CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/malek/tiramisu/benchmarks/halide/wrapper_filter2D.cpp -o CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.s

CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.requires:

.PHONY : CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.requires

CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.provides: CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.requires
	$(MAKE) -f CMakeFiles/bench_filter2D.dir/build.make CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.provides.build
.PHONY : CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.provides

CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.provides.build: CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o


# Object files for target bench_filter2D
bench_filter2D_OBJECTS = \
"CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o"

# External object files for target bench_filter2D
bench_filter2D_EXTERNAL_OBJECTS = \
"/home/malek/tiramisu/build/generated_fct_filter2D.o" \
"/home/malek/tiramisu/build/generated_fct_filter2D_ref.o"

../build/bench_filter2D: CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o
../build/bench_filter2D: ../build/generated_fct_filter2D.o
../build/bench_filter2D: ../build/generated_fct_filter2D_ref.o
../build/bench_filter2D: CMakeFiles/bench_filter2D.dir/build.make
../build/bench_filter2D: ../build/libtiramisu.so
../build/bench_filter2D: ../Halide/lib/libHalide.a
../build/bench_filter2D: ../3rdParty/isl/build/lib/libisl.so
../build/bench_filter2D: CMakeFiles/bench_filter2D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../build/bench_filter2D"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_filter2D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bench_filter2D.dir/build: ../build/bench_filter2D

.PHONY : CMakeFiles/bench_filter2D.dir/build

CMakeFiles/bench_filter2D.dir/requires: CMakeFiles/bench_filter2D.dir/benchmarks/halide/wrapper_filter2D.cpp.o.requires

.PHONY : CMakeFiles/bench_filter2D.dir/requires

CMakeFiles/bench_filter2D.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bench_filter2D.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bench_filter2D.dir/clean

CMakeFiles/bench_filter2D.dir/depend: ../build/generated_fct_filter2D.o
CMakeFiles/bench_filter2D.dir/depend: ../build/generated_fct_filter2D_ref.o
	cd /home/malek/tiramisu/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/malek/tiramisu /home/malek/tiramisu /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug/CMakeFiles/bench_filter2D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bench_filter2D.dir/depend


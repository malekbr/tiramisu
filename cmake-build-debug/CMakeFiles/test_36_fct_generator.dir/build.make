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
include CMakeFiles/test_36_fct_generator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_36_fct_generator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_36_fct_generator.dir/flags.make

CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o: CMakeFiles/test_36_fct_generator.dir/flags.make
CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o: ../tests/test_36.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o -c /home/malek/tiramisu/tests/test_36.cpp

CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/malek/tiramisu/tests/test_36.cpp > CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.i

CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/malek/tiramisu/tests/test_36.cpp -o CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.s

CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.requires:

.PHONY : CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.requires

CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.provides: CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_36_fct_generator.dir/build.make CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.provides.build
.PHONY : CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.provides

CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.provides.build: CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o


# Object files for target test_36_fct_generator
test_36_fct_generator_OBJECTS = \
"CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o"

# External object files for target test_36_fct_generator
test_36_fct_generator_EXTERNAL_OBJECTS =

../build/test_36_fct_generator: CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o
../build/test_36_fct_generator: CMakeFiles/test_36_fct_generator.dir/build.make
../build/test_36_fct_generator: ../build/libtiramisu.so
../build/test_36_fct_generator: ../Halide/lib/libHalide.a
../build/test_36_fct_generator: ../3rdParty/isl/build/lib/libisl.so
../build/test_36_fct_generator: CMakeFiles/test_36_fct_generator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../build/test_36_fct_generator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_36_fct_generator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_36_fct_generator.dir/build: ../build/test_36_fct_generator

.PHONY : CMakeFiles/test_36_fct_generator.dir/build

CMakeFiles/test_36_fct_generator.dir/requires: CMakeFiles/test_36_fct_generator.dir/tests/test_36.cpp.o.requires

.PHONY : CMakeFiles/test_36_fct_generator.dir/requires

CMakeFiles/test_36_fct_generator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_36_fct_generator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_36_fct_generator.dir/clean

CMakeFiles/test_36_fct_generator.dir/depend:
	cd /home/malek/tiramisu/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/malek/tiramisu /home/malek/tiramisu /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug/CMakeFiles/test_36_fct_generator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_36_fct_generator.dir/depend


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
include CMakeFiles/test_22.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_22.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_22.dir/flags.make

../build/generated_fct_test_22.o:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../build/generated_fct_test_22.o"
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/test_22_fct_generator

CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o: CMakeFiles/test_22.dir/flags.make
CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o: ../tests/wrapper_test_22.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o -c /home/malek/tiramisu/tests/wrapper_test_22.cpp

CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/malek/tiramisu/tests/wrapper_test_22.cpp > CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.i

CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/malek/tiramisu/tests/wrapper_test_22.cpp -o CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.s

CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.requires:

.PHONY : CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.requires

CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.provides: CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_22.dir/build.make CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.provides.build
.PHONY : CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.provides

CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.provides.build: CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o


# Object files for target test_22
test_22_OBJECTS = \
"CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o"

# External object files for target test_22
test_22_EXTERNAL_OBJECTS = \
"/home/malek/tiramisu/build/generated_fct_test_22.o"

../build/test_22: CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o
../build/test_22: ../build/generated_fct_test_22.o
../build/test_22: CMakeFiles/test_22.dir/build.make
../build/test_22: ../build/libtiramisu.so
../build/test_22: ../Halide/lib/libHalide.a
../build/test_22: ../3rdParty/isl/build/lib/libisl.so
../build/test_22: CMakeFiles/test_22.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/malek/tiramisu/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../build/test_22"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_22.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_22.dir/build: ../build/test_22

.PHONY : CMakeFiles/test_22.dir/build

CMakeFiles/test_22.dir/requires: CMakeFiles/test_22.dir/tests/wrapper_test_22.cpp.o.requires

.PHONY : CMakeFiles/test_22.dir/requires

CMakeFiles/test_22.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_22.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_22.dir/clean

CMakeFiles/test_22.dir/depend: ../build/generated_fct_test_22.o
	cd /home/malek/tiramisu/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/malek/tiramisu /home/malek/tiramisu /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug/CMakeFiles/test_22.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_22.dir/depend


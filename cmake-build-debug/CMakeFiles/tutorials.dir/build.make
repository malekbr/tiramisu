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

# Utility rule file for tutorials.

# Include the progress variables for this target.
include CMakeFiles/tutorials.dir/progress.make

tutorials: CMakeFiles/tutorials.dir/build.make
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_01
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_02
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_03
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_05
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_06
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_08
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_09
	cd /home/malek/tiramisu && /home/malek/tiramisu/build/tutorial_10
.PHONY : tutorials

# Rule to build all files generated by this target.
CMakeFiles/tutorials.dir/build: tutorials

.PHONY : CMakeFiles/tutorials.dir/build

CMakeFiles/tutorials.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tutorials.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tutorials.dir/clean

CMakeFiles/tutorials.dir/depend:
	cd /home/malek/tiramisu/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/malek/tiramisu /home/malek/tiramisu /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug /home/malek/tiramisu/cmake-build-debug/CMakeFiles/tutorials.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tutorials.dir/depend


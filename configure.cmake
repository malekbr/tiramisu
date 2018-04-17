# Comment out this message if you are done configuring this file
# set(NEEDS_CONFIG true)

if (${NEEDS_CONFIG})
    message(WARNING "Please make sure to configure configure.cmake, and then comment out the second line.")
endif()

set(LLVM_CONFIG_BIN "")

set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/")
set(HALIDE_SOURCE_DIRECTORY "Halide")
set(HALIDE_LIB_DIRECTORY "Halide/lib")

set(MKL_PREFIX "")

set(MPI_PREFIX "")

# Uncomment if you wish to use GPU
set(USE_GPU true)

if(SCRIPT_MODE)
    message("ISL_INCLUDE_DIRECTORY=\"${ISL_INCLUDE_DIRECTORY}\"")
    message("ISL_LIB_DIRECTORY=\"${ISL_LIB_DIRECTORY}\"")
    message("HALIDE_SOURCE_DIRECTORY=\"${HALIDE_SOURCE_DIRECTORY}\"")
    message("HALIDE_LIB_DIRECTORY=\"${HALIDE_LIB_DIRECTORY}\"")
    message("MKL_PREFIX=\"${MKL_PREFIX}\"")
    message("MPI_PREFIX=\"${MPI_PREFIX}\"")
    message("USE_GPU=\"${USE_GPU}\"")
endif()


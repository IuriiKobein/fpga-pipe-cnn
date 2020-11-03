cmake_minimum_required(VERSION 3.0)
project(pipe-cnn)

#quick and dirty cmake - NOT DO IN SUCH WAY
#good guys export cmake tragets and use only targets

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
find_package(IntelFPGAOpenCL )

set(HLSLIB_BOARD_NAME "de10_nano_sharedonly" CACHE STRING "Board name for aoc.")

include_directories(SYSTEM ${IntelFPGAOpenCL_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#so brutal
set(BOARD_SYSROOT "/home/rnd/de10_nano_rootfs")
include_directories(${BOARD_SYSROOT}/usr/local/include)
##find_library(OPENCV_CORE_LIBRARY
##    NAMES opencv_core
##    HINTS "${BOARD_SYSROOT}/usr/local/lib"
##)
##
##find_library(OPENCV_IMGCODECS_LIBRARY
##    NAMES opencv_imgcodecs
##    HINTS "${BOARD_SYSROOT}/usr/local/lib"
##)
##
##find_library(OPENCV_IMGPROC_LIBRARY
##    NAMES opencv_imgproc
##    HINTS "${BOARD_SYSROOT}/usr/local/lib"
##)
##
##find_library(Z_LIBRARY
##    NAMES libz.so
##    HINTS "${BOARD_SYSROOT}/usr/lib/arm-linux-gnueabihf"
##)
##
##find_library(PNG12_LIBRARY
##    NAMES png12
##    HINTS "${BOARD_SYSROOT}/usr/lib/arm-linux-gnueabihf"
##)

set(VEC_SIZE "8")
set(BLOCK_X "256")

set(CMAKE_CXX_FLAGS "-DVEC_SIZE=${VEC_SIZE} ${CMAKE_CXX_FLAGS} -fPIC -std=c++11 --sysroot=${BOARD_SYSROOT} -Wl,--no-as-needed")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(host_pipe_cnn host/pipe_cnn.cpp)
target_link_libraries(host_pipe_cnn ${IntelFPGAOpenCL_LIBRARIES}
    z opencv_core opencv_imgproc opencv_imgcodecs png pthread)

function(opencl_target KERNEL_NAME)
    set(AOCX_NAME ${KERNEL_NAME}_vec${VEC_SIZE}_width${BLOCK_X}.cl)
    set(KERNEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/device/${KERNEL_NAME}.cl)
    set(AOC_COMMAND
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
        -I${CMAKE_SOURCE_DIR}/include
        -fp-relaxed
        -cl-no-signed-zeros
        -cl-fast-relaxed-math
        -cl-single-precision-constant
        -no-interleaving=default
        -board=${HLSLIB_BOARD_NAME}
        -DVEC_SIZE=${VEC_SIZE}
        -DBLOCK_X=${BLOCK_X}
        ${KERNEL_FILE})
    add_custom_target(build_${KERNEL_NAME}_report
        COMMAND ${IntelFPGAOpenCL_AOC}
        ${AOC_COMMAND}
        -report)
    add_custom_target(build_${KERNEL_NAME}_profile
        COMMAND ${IntelFPGAOpenCL_AOC}
        ${AOC_COMMAND}
        -report
        -profile)
    add_custom_target(build_${KERNEL_NAME}_hardware
        COMMAND ${IntelFPGAOpenCL_AOC}
        ${AOC_COMMAND} -o ${AOXC_NAME}_hardware.aocx)
    add_test(Test${KERNEL_NAME} COMMAND host_${KERNEL_NAME} emulator)
endfunction()

opencl_target("conv2d_v1")

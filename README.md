# Piped OpenCL 2D Convolution Kernels on Intel FPGA

The project implement 2d convolution on Intel FPAG(Cyclone V) using pipeline architecture.
Splitting 2d conviolituon in three different stages:
 - reading input from global to local memory
 - calculation partial result
 - writing partila result from local to global memory
allows more efficiency utilize FPGA resource. 

The basic building block in form of 2d convolution with 3x3 kernel
is implemted using OpenCL as HSL for Intel FPGA.

On each iteration new 8 elements vector of input are inserted into hardware shift register by discading the oldest 8 element vector.
Convolution is unrolled to completly flat inner filter convolution

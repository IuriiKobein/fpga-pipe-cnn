set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

#set(CMAKE_SYSROOT /home/rnd/de10_nano_rootfs)
#set(CMAKE_STAGING_PREFIX ${HOME}/stage)

#set(tools /usr/bin/arm-linux-gnueabihf-gcc-ranlib-5)
set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc-5)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++-5)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

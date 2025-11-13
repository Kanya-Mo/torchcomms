# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extension: torchcomms._comms_xccl
file(GLOB TORCHCOMMS_XCCL_SOURCES "comms/torchcomms/xccl/*.cpp")
# find_package(SYCLToolkit REQUIRED)

set(CONDA_INCLUDE "${CONDA_PREFIX}/include")

# find SYCL
set(SYCL_HOME $ENV{SYCL_HOME})
if(NOT SYCL_HOME)
    set(SYCL_HOME "/opt/intel/oneapi/2025.2")
endif()
set(SYCL_INCLUDE "${SYCL_HOME}/include")
message(STATUS "SYCL_INCLUDE is set to ${SYCL_INCLUDE}")

set(XCCL_HOME $ENV{XCCL_HOME})
set(XCCL_INCLUDE "${XCCL_HOME}/include")
message(STATUS "XCCL_INCLUDE is set to ${XCCL_INCLUDE}")
set(XCCL_SHARED_LIB ${XCCL_HOME}/lib/libccl.so.2)
message(STATUS "XCCL_SHARED_LIB is set to ${XCCL_SHARED_LIB}")

add_library(torchcomms_comms_xccl MODULE ${TORCHCOMMS_XCCL_SOURCES})
set_target_properties(torchcomms_comms_xccl PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_xccl"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)

target_include_directories(torchcomms_comms_xccl PRIVATE
    ${ROOT}
    ${XCCL_INCLUDE}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${SYCL_INCLUDE}
    ${SYCL_INCLUDE}/sycl
)
target_compile_features(torchcomms_comms_xccl PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms_xccl PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_xccl PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
)
target_link_libraries(torchcomms_comms_xccl PRIVATE
    ${XCCL_SHARED_LIB}
)

# set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} -std=c++20")

install(TARGETS torchcomms_comms_xccl
        LIBRARY DESTINATION .
)

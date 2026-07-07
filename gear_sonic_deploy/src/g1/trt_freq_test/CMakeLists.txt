set(EXECUTABLE_NAME trt_freq_test)

add_executable(${EXECUTABLE_NAME} src/trt_freq_test.cpp)

set(CUDA_INCLUDES "")
if(CUDAToolkit_FOUND)
  set(CUDA_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
elseif(CUDA_INCLUDE_DIR)
  set(CUDA_INCLUDES ${CUDA_INCLUDE_DIR})
endif()

target_include_directories(
  ${EXECUTABLE_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/g1/g1_deploy_onnx_ref/include
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA_INCLUDES}
)

target_link_libraries(
  ${EXECUTABLE_NAME}
  PRIVATE
    TRTInference
    CUDA::cudart
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
)

set_target_properties(
  ${EXECUTABLE_NAME}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
             OUTPUT_NAME ${EXECUTABLE_NAME}
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

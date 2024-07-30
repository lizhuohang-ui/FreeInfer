cmake_minimum_required(VERSION 3.22.0)
project(free-infer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CTest)
enable_testing()

set(GTest_DIR /usr/local/lib/cmake/GTest)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

aux_source_directory(./src/tensor DIR_SRC_TENSOR)
aux_source_directory(./src/pnnx DIR_SRC_PNNX)
aux_source_directory(./src/runtime DIR_SRC_RUNTIME)
aux_source_directory(./src/layer/layer_factory DIR_SRC_LAYER_FACTORY)
aux_source_directory(./src/layer/layer_activiation DIR_SRC_LAYER_ACTIVIATION)
aux_source_directory(./src/layer/layer_convolution DIR_SRC_LAYER_CONVOLUTION)
aux_source_directory(./src/layer/layer_pooling DIR_SRC_LAYER_POOLING)
aux_source_directory(./src/layer/layer_expression DIR_SRC_LAYER_EXPRESSION)
aux_source_directory(./test DIR_TEST_ARMA)

add_executable(free-infer main.cpp ${DIR_SRC_TENSOR} ${DIR_SRC_PNNX} ${DIR_TEST_ARMA} 
        ${DIR_SRC_RUNTIME} ${DIR_SRC_LAYER_FACTORY} ${DIR_SRC_LAYER_ACTIVIATION} ${DIR_SRC_LAYER_CONVOLUTION}
        ${DIR_SRC_LAYER_POOLING} ${DIR_SRC_LAYER_EXPRESSION})

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${Armadillo_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${GTest_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/tensor
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/pnnx
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/runtime
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/layer/layer_factory
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/layer/layer_activiation
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/layer/layer_convolution
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/layer/layer_pooling
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/include/layer/layer_expression
                                                  )
target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES} ${glog_LIBRARIES} ${link_lib})

message(STATUS ${DIR_SRC_LAYER_ACTIVIATION})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

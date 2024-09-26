cmake_minimum_required(VERSION 3.22.0)
project(free_infer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(fmt REQUIRED)

set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")
find_package(OpenCV REQUIRED)

include_directories(./include)

aux_source_directory(./src/tensor DIR_SRC_TENSOR)
aux_source_directory(./src/pnnx DIR_SRC_PNNX)
aux_source_directory(./src/runtime DIR_SRC_RUNTIME)
aux_source_directory(./src/layer DIR_SRC_LAYER)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(free_infer ${DIR_SRC_TENSOR} ${DIR_SRC_PNNX} ${DIR_TEST_ARMA} 
        ${DIR_SRC_RUNTIME} ${DIR_SRC_LAYER})
    
set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()


target_include_directories(${PROJECT_NAME} PUBLIC ${Armadillo_INCLUDE_DIR} ${glog_INCLUDE_DIR} 
                    ${GTest_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})


target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES} ${link_lib} ${OpenCV_LIBS} fmt::fmt)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

option(DEVELOPMENT "Enable test and bench" ON)
if (${DEVELOPMENT})
    message(STATUS "DEVELOPMENT MODE ON")
    enable_testing()
    add_subdirectory(test)
endif()



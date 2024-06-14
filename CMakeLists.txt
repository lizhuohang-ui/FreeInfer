cmake_minimum_required(VERSION 3.22.0)
project(free-infer VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(GTest_DIR /usr/local/lib/cmake/GTest)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

aux_source_directory(./src/tensor DIR_SRC)
aux_source_directory(./test DIR_TEST_ARMA)

message("=====>" ${DIR_SRC})

add_executable(free-infer main.cpp ${DIR_SRC} ${DIR_TEST_ARMA})

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()


target_include_directories(${PROJECT_NAME} PUBLIC ${Armadillo_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${GTest_INCLUDE_DIR} src/tensor/)
target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES} ${glog_LIBRARIES} ${link_lib})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

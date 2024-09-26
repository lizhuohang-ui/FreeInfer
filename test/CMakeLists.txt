set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)

aux_source_directory(../test/test_layer DIR_TEST_LAYER)
aux_source_directory(../test/test_runtime DIR_TEST_RUNTIME)
aux_source_directory(../test/test_data DIR_TEST_DATA)
# aux_source_directory(../test/test_model DIR_TEST_MODEL)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()

set(link_math_lib ${ARMADILLO_LIBRARIES})

message(STATUS ${DIR_TEST_LAYER})

add_executable(test_freeinfer test_main.cpp ${DIR_TEST_DATA} 
                    ${DIR_TEST_LAYER} ${DIR_TEST_RUNTIME} ${DIR_TEST_MODEL})
                
target_link_directories(test_freeinfer PUBLIC ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(test_freeinfer ${link_lib} ${link_math_lib})
target_link_libraries(test_freeinfer free_infer)

target_include_directories(test_freeinfer PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_freeinfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_freeinfer PUBLIC ${Armadillo_INCLUDE_DIR})

set_target_properties(test_freeinfer PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)


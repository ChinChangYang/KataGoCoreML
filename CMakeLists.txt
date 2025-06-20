cmake_minimum_required(VERSION 3.16)
project(KataGoCoreML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Build unit tests" ON)

# Paths for CoreMLTools ML model
set(COREMLTOOLS_BUILD_MLMODEL   ${CMAKE_SOURCE_DIR}/coremltools/build/mlmodel)
set(COREMLTOOLS_INCLUDE_MLMODEL ${CMAKE_SOURCE_DIR}/coremltools/mlmodel/src)

# Paths for CoreMLTools ML model format
set(COREMLTOOLS_INCLUDE_MLFORMAT ${CMAKE_SOURCE_DIR}/coremltools/mlmodel/build/format)

# Paths for CoreMLTools ML package
set(COREMLTOOLS_BUILD_MLPACKAGE   ${CMAKE_SOURCE_DIR}/coremltools/build)
set(COREMLTOOLS_INCLUDE_MLPACKAGE ${CMAKE_SOURCE_DIR}/coremltools/modelpackage/src)

# Paths for CoreMLTools protobuf
set(PROTOBUF_SRC_DIR ${CMAKE_SOURCE_DIR}/coremltools/deps/protobuf/src)
set(PROTOBUF_LIB_DIR ${CMAKE_SOURCE_DIR}/coremltools/build/deps/protobuf/cmake)

# Gather sources files
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# KataGoCoreML library
add_library(katagocoreml SHARED ${SRC_FILES})

# Public include: your API headers
target_include_directories(katagocoreml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Private includes
target_include_directories(katagocoreml
    PRIVATE
        ${COREMLTOOLS_INCLUDE_MLMODEL}
        ${COREMLTOOLS_INCLUDE_MLFORMAT}
        ${COREMLTOOLS_INCLUDE_MLPACKAGE}
        ${PROTOBUF_SRC_DIR}
)

# Link directories for the libraries manually built by scripts/build_coremltools.sh
target_link_directories(katagocoreml
    PRIVATE
        ${COREMLTOOLS_BUILD_MLMODEL}
        ${COREMLTOOLS_BUILD_MLPACKAGE}
        ${PROTOBUF_LIB_DIR}
)

# Find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Link libraries for the libraries manually built by scripts/build_coremltools.sh:
#   - mlmodel: CoreMLTools ML model (static)
#   - modelpackage: CoreMLTools ML package (shared)
#   - protobuf: CoreMLTools protobuf (static)
#   - Python3: Python3 interpreter and development (shared)
target_link_libraries(katagocoreml
    PUBLIC
        mlmodel
        protobuf
    PRIVATE
        modelpackage
        ${Python3_LIBRARIES}
)

# Check if the CoreMLTools ML package shared library exists
# and if not, invoke the build_coremltools.sh script to build it.
set(MODELPACKAGE_LIB "${COREMLTOOLS_BUILD_MLPACKAGE}/libmodelpackage.dylib")
if(NOT EXISTS ${MODELPACKAGE_LIB})
    message(STATUS "libmodelpackage.dylib not found; invoking build_coremltools.sh")
    execute_process(
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/build_coremltools.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE coremltools_build_result
    )
    if(NOT coremltools_build_result EQUAL 0 OR NOT EXISTS ${MODELPACKAGE_LIB})
        message(FATAL_ERROR
            "Failed to build CoreMLTools ML package. "
            "Please check scripts/build_coremltools.sh"
        )
    endif()
endif()

# Install CoreMLTools ML package shared library
install(
    FILES ${COREMLTOOLS_BUILD_MLPACKAGE}/libmodelpackage.dylib
    DESTINATION lib
)

# Install KataGoCoreML library
install(TARGETS katagocoreml EXPORT KataGoCoreMLTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install KataGoCoreML headers
install(DIRECTORY include/ DESTINATION include)

# Install KataGoCoreML CMake configuration
install(EXPORT KataGoCoreMLTargets
    FILE KataGoCoreMLConfig.cmake
    NAMESPACE KataGoCoreML::
    DESTINATION lib/cmake/KataGoCoreML
)

# Uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)

# Testing
if(BUILD_TESTING)
    enable_testing()
    add_executable(katagocoreml_tests test/test_main.cpp)
    target_link_directories(katagocoreml_tests
        PRIVATE
            ${COREMLTOOLS_BUILD_MLMODEL}
            ${PROTOBUF_LIB_DIR}
    )
    target_link_libraries(katagocoreml_tests PRIVATE katagocoreml)
    add_test(NAME BasicTest COMMAND katagocoreml_tests)
endif()

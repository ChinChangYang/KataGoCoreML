cmake_minimum_required(VERSION 3.16)
project(KataGoCoreML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared instead of static libraries" OFF)
option(BUILD_TESTING "Build unit tests" ON)

# — Paths for CoreMLTools native model builder —
set(COREMLTOOLS_BUILD_MLMODEL ${CMAKE_SOURCE_DIR}/coremltools/build/mlmodel)
set(COREMLTOOLS_INCLUDE_MLMODEL ${COREMLTOOLS_BUILD_MLMODEL}/format)

# - Paths for CoreMLTools ML package -
set(COREMLTOOLS_BUILD_MLPACKAGE ${CMAKE_SOURCE_DIR}/coremltools/build)
set(COREMLTOOLS_INCLUDE_MLPACKAGE ${CMAKE_SOURCE_DIR}/coremltools/modelpackage/src)

# — Paths for CoreMLTools’ protobuf —
set(PROTOBUF_SRC_DIR   ${CMAKE_SOURCE_DIR}/coremltools/deps/protobuf/src)
set(PROTOBUF_LIB_DIR   ${CMAKE_SOURCE_DIR}/coremltools/build/deps/protobuf/cmake)

# — Gather your sources —
file(GLOB_RECURSE SRC_FILES src/*.cpp)

add_library(katagocoreml SHARED ${SRC_FILES})

# — Public include: your API headers —
#    BUILD_INTERFACE: during build use full source path
#    INSTALL_INTERFACE: after install use "include"
target_include_directories(katagocoreml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# — Private includes —
target_include_directories(katagocoreml
    PRIVATE
        ${COREMLTOOLS_INCLUDE_MLMODEL}
        ${COREMLTOOLS_INCLUDE_MLPACKAGE}
        ${PROTOBUF_SRC_DIR}
)

# — Link directories —
target_link_directories(katagocoreml
    PRIVATE
        ${COREMLTOOLS_BUILD_MLMODEL}
        ${COREMLTOOLS_BUILD_MLPACKAGE}
        ${PROTOBUF_LIB_DIR}
)

# - Find Python3 -
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# — Link libraries —
target_link_libraries(katagocoreml
    PUBLIC
        mlmodel
        protobuf
    PRIVATE
        modelpackage
        ${Python3_LIBRARIES}
)

# — Installation / Export —
install(TARGETS katagocoreml EXPORT KataGoCoreMLTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT KataGoCoreMLTargets
    FILE KataGoCoreMLConfig.cmake
    NAMESPACE KataGoCoreML::
    DESTINATION lib/cmake/KataGoCoreML
)

# — Testing —
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

cmake_minimum_required(VERSION 3.20)
project(tatar_ocr_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TATAR_OCR_BUILD_PYTHON "Build pybind11 Python extension module" ON)

set(NATIVE_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.native_deps")

# Locate OpenCV (supports explicit OpenCV_DIR or .native_deps)
if(NOT OpenCV_DIR AND EXISTS "${NATIVE_DEPS_DIR}")
    file(GLOB _ocv_candidates "${NATIVE_DEPS_DIR}/opencv/build" "${NATIVE_DEPS_DIR}/opencv*/build" "${NATIVE_DEPS_DIR}/opencv*")
    foreach(_cand IN LISTS _ocv_candidates)
        if(EXISTS "${_cand}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${_cand}")
            break()
        endif()
    endforeach()
endif()
find_package(OpenCV REQUIRED)

# Locate nlohmann/json
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    find_path(NLOHMANN_JSON_INCLUDE_DIR
        NAMES nlohmann/json.hpp
        HINTS
            "${nlohmann_json_DIR}"
            "${NATIVE_DEPS_DIR}/nlohmann_json/include"
            "${NATIVE_DEPS_DIR}/nlohmann/include"
            "${NATIVE_DEPS_DIR}/include"
            "${NATIVE_DEPS_DIR}"
    )
    if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json.hpp not found. Set nlohmann_json_DIR or NLOHMANN_JSON_INCLUDE_DIR.")
    endif()
endif()

# Locate ONNX Runtime C++ SDK
if(NOT ONNXRUNTIME_ROOT AND EXISTS "${NATIVE_DEPS_DIR}")
    file(GLOB _ort_candidates "${NATIVE_DEPS_DIR}/onnxruntime*")
    foreach(_cand IN LISTS _ort_candidates)
        if(IS_DIRECTORY "${_cand}")
            set(ONNXRUNTIME_ROOT "${_cand}")
            break()
        endif()
    endforeach()
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/build/native/include"
    PATH_SUFFIXES onnxruntime
)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native"
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime C++ SDK not found. Set ONNXRUNTIME_ROOT.")
endif()

add_library(tatar_ocr_core STATIC src/tatar_ocr.cpp)
set_target_properties(tatar_ocr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tatar_ocr_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
        "${ONNXRUNTIME_INCLUDE_DIR}"
)
target_link_libraries(tatar_ocr_core PUBLIC ${OpenCV_LIBS} "${ONNXRUNTIME_LIB}")

if(nlohmann_json_FOUND)
    target_link_libraries(tatar_ocr_core PUBLIC nlohmann_json::nlohmann_json)
else()
    target_include_directories(tatar_ocr_core PUBLIC "${NLOHMANN_JSON_INCLUDE_DIR}")
endif()

if(MSVC)
    target_compile_options(tatar_ocr_core PUBLIC /utf-8 /W3)
else()
    target_compile_options(tatar_ocr_core PUBLIC -Wall -Wextra)
endif()

if(TATAR_OCR_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_pybind11_cmake_dir)
            set(pybind11_DIR "${_pybind11_cmake_dir}")
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(FATAL_ERROR "pybind11 not found. Install pybind11 or set pybind11_DIR.")
        endif()
    endif()

    pybind11_add_module(tatar_ocr_native src/py_module.cpp)
    target_link_libraries(tatar_ocr_native PRIVATE tatar_ocr_core)
    set_target_properties(tatar_ocr_native PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
    )

    if(WIN32)
        file(GLOB _ort_dlls "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.dll" "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime*.dll")
        file(GLOB _ocv_dlls "${OpenCV_DIR}/x64/vc*/bin/opencv_world*.dll" "${OpenCV_DIR}/bin/opencv_world*.dll")
        foreach(_dll IN LISTS _ort_dlls _ocv_dlls)
            add_custom_command(TARGET tatar_ocr_native POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:tatar_ocr_native>"
            )
        endforeach()
    endif()
endif()

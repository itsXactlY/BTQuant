cmake_minimum_required(VERSION 3.10)
project(hotspine)

add_library(hotspine_writer STATIC
    hotspine_writer.cpp
)

add_library(hotspine_reader STATIC
    hotspine_reader.cpp
    hotspine_reader_c_interface.cpp
)

target_include_directories(hotspine_writer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../market_data_collector
)

target_include_directories(hotspine_reader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hotspine_writer
    PRIVATE
        rt
)

target_link_libraries(hotspine_reader
    PRIVATE
        rt
)

# Create shared library for Python bindings
add_library(hotspine_reader_shared SHARED
    hotspine_reader.cpp
    hotspine_reader_c_interface.cpp
)

target_include_directories(hotspine_reader_shared PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hotspine_reader_shared
    PRIVATE
        rt
)

# Set output name for shared library
set_target_properties(hotspine_reader_shared PROPERTIES
    OUTPUT_NAME "hotspine_reader"
    PREFIX "lib"
    SUFFIX ".so"
)
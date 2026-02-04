cmake_minimum_required(VERSION 3.10)
project(PubBTQuantExamples)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Define include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../dependencies/BTQ_Render_Engine/include)

# Define source files for the TaskScheduler
set(TASK_SCHEDULER_SOURCES
    ../dependencies/BTQ_Render_Engine/src/threading/task_scheduler.cpp
)

# Create executables for each example
add_executable(volume_analysis_example volume_analysis_example.cpp ${TASK_SCHEDULER_SOURCES})
target_link_libraries(volume_analysis_example Threads::Threads)

add_executable(indicator_calculations_example indicator_calculations_example.cpp ${TASK_SCHEDULER_SOURCES})
target_link_libraries(indicator_calculations_example Threads::Threads)

add_executable(data_processing_example data_processing_example.cpp ${TASK_SCHEDULER_SOURCES})
target_link_libraries(data_processing_example Threads::Threads)

add_executable(correlation_analytics_example correlation_analytics_example.cpp ${TASK_SCHEDULER_SOURCES})
target_link_libraries(correlation_analytics_example Threads::Threads)

# Set compiler flags for better debugging and optimization
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(volume_analysis_example PRIVATE -g -Wall -Wextra)
    target_compile_options(indicator_calculations_example PRIVATE -g -Wall -Wextra)
    target_compile_options(data_processing_example PRIVATE -g -Wall -Wextra)
    target_compile_options(correlation_analytics_example PRIVATE -g -Wall -Wextra)
else()
    target_compile_options(volume_analysis_example PRIVATE -O2 -Wall)
    target_compile_options(indicator_calculations_example PRIVATE -O2 -Wall)
    target_compile_options(data_processing_example PRIVATE -O2 -Wall)
    target_compile_options(correlation_analytics_example PRIVATE -O2 -Wall)
endif()
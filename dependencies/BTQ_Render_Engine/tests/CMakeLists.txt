cmake_minimum_required(VERSION 3.10)
project(BTQ_Render_Engine_Tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Vulkan REQUIRED)

# Find GLM
find_path(GLM_INCLUDE_DIR glm/glm.hpp)
if(NOT GLM_INCLUDE_DIR)
    # If not found via package manager, try common installation paths
    find_path(GLM_INCLUDE_DIR glm/glm.hpp
        PATHS /usr/include /usr/local/include /opt/local/include
        NO_DEFAULT_PATH
    )
endif()

# Find nlohmann json
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    # If not found via package manager, try common installation paths
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS /usr/include /usr/local/include /opt/local/include
        NO_DEFAULT_PATH
    )
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/components
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/data
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/layout
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/system
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/trading
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/ui
)

# Add GLM include if found
if(GLM_INCLUDE_DIR)
    include_directories(${GLM_INCLUDE_DIR})
    add_definitions(-DUSE_GLM)
else()
    message(WARNING "GLM not found, some features may not work")
endif()

# Add nlohmann json include if found
if(NLOHMANN_JSON_INCLUDE_DIR)
    include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
    add_definitions(-DHAS_NLOHMANN_JSON)
else()
    message(WARNING "nlohmann/json not found, some features may not work")
endif()

# Define source files for the test
set(TEST_SOURCES
    ../src/ui/animation_system.cpp
    ../src/ui/unified_theme_system.cpp
    ../src/system/memory_optimizer.cpp
    ../src/layout/layout_presets.cpp
    ../src/layout/dashboard_layout_manager.cpp
    ../src/components/theme_manager.cpp
    optimization_tests.cpp
)

# Set properties for imgui if available
file(GLOB IMGUI_SOURCES "../src/imgui/*.cpp")
if(IMGUI_SOURCES)
    list(APPEND TEST_SOURCES ${IMGUI_SOURCES})
endif()

# Add executable
add_executable(optimization_tests ${TEST_SOURCES})

# Include directories
target_include_directories(optimization_tests PRIVATE ../src/imgui)

# Link libraries
target_link_libraries(optimization_tests
    ${VULKAN_LIBRARY}
)

# Set properties for imgui if available
file(GLOB IMGUI_SOURCES "../src/imgui/*.cpp")
if(IMGUI_SOURCES)
    list(APPEND TEST_SOURCES ${IMGUI_SOURCES})
    target_include_directories(optimization_tests PRIVATE ../src/imgui)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(optimization_tests PRIVATE -Wall -Wextra -pedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(optimization_tests PRIVATE /W4)
endif()
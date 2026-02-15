cmake_minimum_required(VERSION 3.15)
project(PubBTQuant LANGUAGES CXX C)

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global build settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type option
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address,undefined")

# Option to build tests
option(BUILD_TESTS "Build test executables" OFF)
option(BUILD_DOCS "Build documentation" OFF)

# Define source files for the main project
set(PUBBTQUANT_SOURCES
    src/main.cpp
    src/analytics/tpoengine.cpp
    src/analytics/liquiditysweepdetector.cpp
    src/analytics/rawtradetable.cpp
    src/ui/compute_to_imgui_bind.cpp
    src/ui/ssbo_aggregator.cpp
    src/audio/pitch_shifter.cpp
    src/audio/audio_engine_integration.cpp
    dependencies/BTQ_Render_Engine/src/trading/trade_command_queue.cpp
)

# Find Dependencies
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# Add subdirectories
add_subdirectory(dependencies/BTQ_Render_Engine)

# Create main executable if there are sources
if(PUBBTQUANT_SOURCES)
    add_executable(pubbtquant ${PUBBTQUANT_SOURCES})

    # Include directories
    target_include_directories(pubbtquant PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ui
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/BTQ_Render_Engine/include
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/BTQ_Render_Engine/src/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/BTQ_Render_Engine/_deps/imgui-src
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/BTQ_Render_Engine/_deps/imgui-src/backends
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/BTQ_Render_Engine/_deps/implot-src
        ${CMAKE_CURRENT_SOURCE_DIR}/_deps/readerwriterqueue-src
        ${CMAKE_CURRENT_SOURCE_DIR}/_deps/miniaudio-src
    )

    # Define preprocessor definitions
    target_compile_definitions(pubbtquant PRIVATE
        MINIAUDIO_IMPLEMENTATION
    )

    # Link against the necessary libraries
    target_link_libraries(pubbtquant PRIVATE
        btq_render_lib
        glfw
        dl
        pthread
    )

endif()

# Documentation target
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_SOURCE_DIR}/Doxyfile ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    endif()
endif()

# Installation rules
include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/shaders/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/pubbtquant/shaders
    COMPONENT runtime
    FILES_MATCHING PATTERN "*"
)

install(FILES ${CMAKE_SOURCE_DIR}/default_layout.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/pubbtquant
    COMPONENT runtime
)

# Package configuration
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
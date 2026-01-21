cmake_minimum_required(VERSION 3.15)
project(BTQ_Render_Engine VERSION 1.0.0 LANGUAGES CXX)

# ============================================
# Compiler Configuration
# ============================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compilation flags for performance
if(MSVC)
    add_compile_options(/W4 /WX /O2 /fp:fast /openmp)
else()
    add_compile_options(-Wall -Wextra -Werror -O3 -ffast-math -march=native -fopenmp)
endif()

# ============================================
# Vulkan SDK Configuration
# ============================================

find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found!")
endif()

message(STATUS "Vulkan SDK Found: ${Vulkan_VERSION}")
message(STATUS "Vulkan Include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Library: ${Vulkan_LIBRARIES}")

# ============================================
# Source Files
# ============================================

set(HEADER_FILES
    include/components/MarketMicrostructureRenderer.h
    include/components/VulkanSynchronization.h
    include/trading/HotspineData.h
    include/MarketMicrostructureRenderer.arch.md
)

set(SOURCE_FILES
    src/components/MarketMicrostructureRenderer.cpp
    src/components/VulkanSynchronization.cpp
)

# ============================================
# Shader Files
# ============================================

add_subdirectory(shaders)

# ============================================
# Library Target
# ============================================

add_library(BTQ_Render_Engine STATIC
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

# ============================================
# Include Directories
# ============================================

target_include_directories(BTQ_Render_Engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/btq_render_engine>
        ${Vulkan_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ============================================
# Link Libraries
# ============================================

target_link_libraries(BTQ_Render_Engine
    PUBLIC
        Vulkan::Vulkan
)

# ============================================
# Target Properties
# ============================================

set_target_properties(BTQ_Render_Engine PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${HEADER_FILES}"
    COMPILE_WARNING_AS_ERROR ON
)

# ============================================
# Installation Configuration
# ============================================

include(GNUInstallDirs)

# Install library
install(TARGETS BTQ_Render_Engine
    EXPORT BTQ_Render_EngineTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/btq_render_engine
)

# Install shaders
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders
    DESTINATION ${CMAKE_INSTALL_DATADIR}/btq_render_engine
)

# CMake export configuration
install(EXPORT BTQ_Render_EngineTargets
    FILE BTQ_Render_EngineTargets.cmake
    NAMESPACE BTQ::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BTQ_Render_Engine
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/BTQ_Render_EngineConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BTQ_Render_EngineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/BTQ_Render_EngineConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BTQ_Render_Engine
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/BTQ_Render_EngineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BTQ_Render_EngineConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BTQ_Render_Engine
)

# ============================================
# Example Target (Optional)
# ============================================

option(BTQ_BUILD_EXAMPLES "Build examples" OFF)

if(BTQ_BUILD_EXAMPLES)
    message(STATUS "Building examples")
    
    add_executable(MarketMicrostructureExample
        examples/MarketMicrostructureExample.cpp
    )
    
    target_link_libraries(MarketMicrostructureExample
        PRIVATE
            BTQ::BTQ_Render_Engine
    )
    
    target_include_directories(MarketMicrostructureExample
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    # Copy shaders to example build directory
    add_custom_command(TARGET MarketMicrostructureExample POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_BINARY_DIR}/shaders
                $<TARGET_FILE_DIR:MarketMicrostructureExample>/shaders
    )
    
    # Install example
    install(TARGETS MarketMicrostructureExample
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# ============================================
# Documentation (Optional)
# ============================================

option(BTQ_BUILD_DOCUMENTATION "Build documentation" OFF)

if(BTQ_BUILD_DOCUMENTATION)
    find_package(Doxygen REQUIRED)
    
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT documentation
    )
endif()

# ============================================
# Testing (Optional)
# ============================================

option(BTQ_BUILD_TESTS "Build tests" OFF)

if(BTQ_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    add_subdirectory(tests)
    
    include(GoogleTest)
    gtest_discover_tests(BTQ_Render_Engine_Tests)
endif()

# ============================================
# Summary
# ============================================

message(STATUS "============================================")
message(STATUS "BTQ Render Engine Configuration")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Examples: ${BTQ_BUILD_EXAMPLES}")
message(STATUS "Documentation: ${BTQ_BUILD_DOCUMENTATION}")
message(STATUS "Tests: ${BTQ_BUILD_TESTS}")
message(STATUS "Vulkan: ${Vulkan_VERSION}")
message(STATUS "============================================")
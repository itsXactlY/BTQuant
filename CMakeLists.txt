cmake_minimum_required(VERSION 3.15)
project(PubBTQuant LANGUAGES CXX C)

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global build settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option to build tests
option(BUILD_TESTS "Build test executables" OFF)
option(BUILD_DOCS "Build documentation" OFF)

# Add subdirectories
add_subdirectory(dependencies/BTQ_Render_Engine)

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
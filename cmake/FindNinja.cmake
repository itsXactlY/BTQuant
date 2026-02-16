# FindNinja.cmake
# Finds the Ninja build system
#
# This module defines:
#   Ninja_FOUND        - True if Ninja is found
#   Ninja_EXECUTABLE   - The Ninja executable path
#   Ninja_VERSION      - The Ninja version string

find_program(Ninja_EXECUTABLE ninja)

if(Ninja_EXECUTABLE)
    execute_process(
        COMMAND ${Ninja_EXECUTABLE} --version
        OUTPUT_VARIABLE Ninja_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(Ninja_VERSION "${Ninja_VERSION_OUTPUT}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ninja
    REQUIRED_VARS Ninja_EXECUTABLE
    VERSION_VAR Ninja_VERSION
)

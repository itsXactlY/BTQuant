cmake_minimum_required(VERSION 3.10)
project(BTQRenderEngine)

set(CMAKE_CXX_STANDARD 17)

# Find Qt5
find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)

# Enable automatic MOC, UI, and RCC processing
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Include directories
include_directories(include)

# Add the TPO engine library
add_library(tpoengine
    src/analytics/tpoengine.cpp
)

# Add the TPO profile panel component
set_source_files_properties(src/components/tpoprofilepanel.cpp PROPERTIES SKIP_AUTOMOC ON)
qt5_wrap_cpp(tpoprofilepanel_moc_srcs include/components/tpoprofilepanel.h)

add_library(components
    src/components/tpoprofilepanel.cpp
    ${tpoprofilepanel_moc_srcs}
)

# Link Qt5 libraries to the components library
target_link_libraries(components Qt5::Core Qt5::Widgets Qt5::Gui)

# Create executables for testing
add_executable(tpotest
    test/tpotest.cpp
)

add_executable(tpoenginetest
    test/tpoengine_test.cpp
)

add_executable(tpoprofilepaneltest
    test/tpoprofilepanel_test.cpp
)

# Link the TPO engine library to the test executables
target_link_libraries(tpotest tpoengine)
target_link_libraries(tpoenginetest tpoengine)

# Link the components library to the TPO profile panel test
target_link_libraries(tpoprofilepaneltest components tpoengine Qt5::Core Qt5::Widgets Qt5::Gui)

# Set output directory for binaries
set_target_properties(tpotest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(tpoprofilepaneltest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
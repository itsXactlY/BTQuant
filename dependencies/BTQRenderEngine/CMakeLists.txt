cmake_minimum_required(VERSION 3.22)
project(BTQRenderEngine LANGUAGES CXX)
set(CMAKE_POLICY_VERSION_MINIMUM 3.22)

# C++26 Standard - Cutting Edge (GCC 15+ / Clang 18+) for main application
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Dependencies
find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# Enable automatic MOC, UI, and RCC processing
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# External Headers
include(FetchContent)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.2)
FetchContent_Declare(concurrentqueue GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git GIT_TAG v1.0.4)

# ImGui Docking Branch
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG docking)

# ImPlot
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG master)

FetchContent_MakeAvailable(glm json concurrentqueue imgui implot)

# Path Configuration
set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMPLOT_DIR ${implot_SOURCE_DIR})

# Include directories
include_directories(include)
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${IMGUI_DIR})
include_directories(${IMGUI_DIR}/backends)
include_directories(${IMPLOT_DIR})

# Add the data management library
add_library(btqdata
    src/data/orderbookhistory.cpp
)

# Add the TPO engine library
add_library(tpoengine
    src/analytics/tpoengine.cpp
)

# Add the TPO profile panel component
set_source_files_properties(src/components/tpoprofilepanel.cpp PROPERTIES SKIP_AUTOMOC ON)
qt5_wrap_cpp(tpoprofilepanel_moc_srcs include/components/tpoprofilepanel.h)

# Add the DOM surface panel component
add_library(components
    src/components/tpoprofilepanel.cpp
    src/components/domsurfacepanel.cpp
    src/components/pricestatisticpanel.cpp
    ${tpoprofilepanel_moc_srcs}
)

# Link Qt5, Vulkan, and ImGui libraries to the components library
target_link_libraries(components Qt5::Core Qt5::Widgets Qt5::Gui ${Vulkan_LIBRARIES} imgui implot)

# Create executables for testing
add_executable(tpotest
    test/tpotest.cpp
)

add_executable(tpoenginetest
    test/tpoengine_test.cpp
)

add_executable(orderbookhistorytest
    test/orderbookhistory_test.cpp
)

add_executable(tpoprofilepaneltest
    test/tpoprofilepanel_test.cpp
)

add_executable(domsurfacepaneltest
    test/domsurfacepanel_test.cpp
)

# Link the TPO engine library to the test executables
target_link_libraries(tpotest tpoengine ${Vulkan_LIBRARIES} imgui implot)
target_link_libraries(tpoenginetest tpoengine ${Vulkan_LIBRARIES} imgui implot)

# Link the data library to the orderbook history test
target_link_libraries(orderbookhistorytest btqdata Threads::Threads)

# Link the components library to the TPO profile panel test
target_link_libraries(tpoprofilepaneltest components tpoengine Qt5::Core Qt5::Widgets Qt5::Gui ${Vulkan_LIBRARIES} imgui implot)

# Link the components library to the DOM surface panel test
target_link_libraries(domsurfacepaneltest components tpoengine Qt5::Core Qt5::Widgets Qt5::Gui ${Vulkan_LIBRARIES} imgui implot)

# Set output directory for binaries
set_target_properties(tpotest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(tpoprofilepaneltest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
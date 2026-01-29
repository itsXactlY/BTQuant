# Install script for directory: /home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/concurrentqueue/concurrentqueueTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/concurrentqueue/concurrentqueueTargets.cmake"
         "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-build/CMakeFiles/Export/a0893b6404edf66f9e8aae21e38f7549/concurrentqueueTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/concurrentqueue/concurrentqueueTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/concurrentqueue/concurrentqueueTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/concurrentqueue" TYPE FILE FILES "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-build/CMakeFiles/Export/a0893b6404edf66f9e8aae21e38f7549/concurrentqueueTargets.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/concurrentqueue" TYPE FILE FILES
    "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-build/concurrentqueueConfig.cmake"
    "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-build/concurrentqueueConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/concurrentqueue/moodycamel" TYPE FILE FILES
    "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-src/blockingconcurrentqueue.h"
    "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-src/concurrentqueue.h"
    "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-src/lightweightsemaphore.h"
    "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-src/LICENSE.md"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/alca/projects/PubBTQuant/.ralphy-worktrees/agent-4-1769694658807-6pne7j/dependencies/BTQ_Render_Engine/build_my_test/_deps/concurrentqueue-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()

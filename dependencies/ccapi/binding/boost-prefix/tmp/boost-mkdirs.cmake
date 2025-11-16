# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost")
  file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost")
endif()
file(MAKE_DIRECTORY
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix/src/boost-build"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix/tmp"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix/src/boost-stamp"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix/src"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix/src/boost-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix/src/boost-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/boost-prefix/src/boost-stamp${cfgdir}") # cfgdir has leading slash
endif()

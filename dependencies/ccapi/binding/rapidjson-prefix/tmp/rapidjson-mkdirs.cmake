# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson")
  file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson")
endif()
file(MAKE_DIRECTORY
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix/src/rapidjson-build"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix/tmp"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix/src/rapidjson-stamp"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix/src"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix/src/rapidjson-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix/src/rapidjson-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/rapidjson-prefix/src/rapidjson-stamp${cfgdir}") # cfgdir has leading slash
endif()

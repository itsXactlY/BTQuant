# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix")
  file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix")
endif()
file(MAKE_DIRECTORY
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix/src/hffix-build"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix/tmp"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix/src/hffix-stamp"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix/src"
  "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix/src/hffix-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix/src/hffix-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/alca/projects/PubBTQuant/dependencies/ccapi/binding/hffix-prefix/src/hffix-stamp${cfgdir}") # cfgdir has leading slash
endif()

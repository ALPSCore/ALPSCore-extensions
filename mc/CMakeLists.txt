# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8.12)

# define project name
project (alps-mc-extensions CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)
# add common definitions
include(ALPSCommonModuleDefinitions)
include(ALPSExtensionDefinitions)

gen_documentation()
if (DocumentationOnly)
  return()
endif ()

add_alpscore(mc)
cxx11()
fix_rpath()
no_source_builds()

add_this_package(mc_metropolis)
#add_alps_package(alps-mc)

add_testing()

gen_cfg_module()

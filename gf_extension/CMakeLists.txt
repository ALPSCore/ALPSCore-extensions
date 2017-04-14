# 
# alps-gf cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8.12)

# define project name
project (alps-gf-extensions CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

find_package(Eigen3 REQUIRED)
find_package(LAPACK REQUIRED)

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)
# add common definitions
include(ALPSCommonModuleDefinitions)
include(ALPSExtensionDefinitions)

gen_documentation()
if (DocumentationOnly)
  return()
endif ()

add_alpscore(gf)
cxx11()
fix_rpath()
no_source_builds()

add_this_package(gf_extension)
target_include_directories(alps-gf-extensions PUBLIC ${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/unsupported)
target_compile_options(alps-gf-extensions PRIVATE ${LAPACK_LINKER_FLAGS})
target_link_libraries(alps-gf-extensions PRIVATE ${LAPACK_LIBRARIES})

add_testing()

gen_cfg_module()

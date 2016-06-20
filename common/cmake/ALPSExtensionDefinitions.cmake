# C++11
macro(cxx11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
        if (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
            message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
        endif ()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
        if ("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
            message( STATUS "Adding -stdlib=libc++ flag")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        endif()
    else ()
        message(FATAL_ERROR "Your C++ compiler does not support C++11.")
    endif ()
endmacro(cxx11)

# Set global rpath
macro(fix_rpath)
    option(FIX_RPATH TRUE "Fix rpath for dynamically linked targets")
    if (FIX_RPATH)
        # RPATH fix
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
            set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
        else()
            set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        endif()

        #policy update CMP0042
        if(APPLE)
            set(CMAKE_MACOSX_RPATH ON)
        endif()
    endif()
endmacro(fix_rpath)



# Disable in-source builds
macro(no_source_builds)
    if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory.")
    endif()
    # Print build directory
    message(STATUS "BUILD_DIR: ${CMAKE_BINARY_DIR}")
    # Print source directory
    message(STATUS "SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
    set(CMAKE_DISABLE_SOURCE_CHANGES ON)
    set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
endmacro(no_source_builds)




# ALPSCore - brings MPI, HDF5, boost 
macro(add_alpscore)
find_package(ALPSCore REQUIRED COMPONENTS hdf5 accumulators mc params)
    message(STATUS "ALPSCore includes:  ${ALPSCore_INCLUDES}")
    message(STATUS "ALPSCore libraries: ${ALPSCore_LIBRARIES}")
    include_directories(${ALPSCore_INCLUDE_DIRS})
    link_libraries(${ALPSCore_LIBRARIES})
endmacro(add_alpscore)



cmake_minimum_required(VERSION 3.16)
project(NumericalMLInC VERSION 1.0.0 LANGUAGES C)

# Global settings-----------------------------------------------------------

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler flags-------------------------------------------------------------

add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options("$<$<CONFIG:Debug>:-g3>")
add_compile_options("$<$<CONFIG:Debug>:-O0>")
add_compile_options("$<$<CONFIG:Release>:-O3>")
add_compile_definitions("$<$<CONFIG:Release>:NDEBUG>")

option(ML_DEBUG_MEMORY "Enable memory tracking/debug" OFF)
if(ML_DEBUG_MEMORY)
    add_compile_definitions(ML_DEBUG_MEMORY)
endif()

# ml_core  —  memory + vector + matrix---------------------------------------

add_library(ml_core STATIC
    src/core/memory.c
    src/core/vector.c
    src/core/Matrix.c
)
target_include_directories(ml_core PUBLIC src/core)
target_link_libraries(ml_core PUBLIC m)

# ml_numerical  —  SVD + gradient + random + tree + argsort------------------

add_library(ml_numerical STATIC
    src/numerical/SVD.c
    src/numerical/gradient.c
    src/numerical/random.c
    src/numerical/tree.c
    src/models/argsort.c      # tree depende de argsort
)
target_include_directories(ml_numerical PUBLIC src/numerical src/models)
target_link_libraries(ml_numerical PUBLIC ml_core)

# ml_models  —  LinearRegression + LogisticRegression + RandomForest---------

add_library(ml_models STATIC
    src/models/LinearRegression.c
    src/models/LogisticRegression.c
    src/models/RandomForest.c
)
target_include_directories(ml_models PUBLIC src/models)
target_link_libraries(ml_models PUBLIC ml_numerical)

# Ejecutables de ejemplo-----------------------------------------------------

foreach(example LinearTest LogisticTest RandomForestTest)
    add_executable(${example} examples/${example}.c)
    target_include_directories(${example} PRIVATE src/core src/models src/numerical)
    target_link_libraries(${example} PRIVATE ml_models)
endforeach()

# Reglas de instalación------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS ml_core ml_numerical ml_models
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY src/core/      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ml/core      FILES_MATCHING PATTERN "*.h")
install(DIRECTORY src/numerical/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ml/numerical FILES_MATCHING PATTERN "*.h")
install(DIRECTORY src/models/    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ml/models    FILES_MATCHING PATTERN "*.h")

# Summary--------------------------------------------------------------------

message(STATUS "")
message(STATUS "=== Numerical-ML-in-C ===")
message(STATUS "  Libs     : ml_core  ml_numerical  ml_models")
message(STATUS "  Examples : LinearTest  LogisticTest  RandomForestTest")
message(STATUS "  Build    : cmake -B build && cmake --build build")
message(STATUS "")

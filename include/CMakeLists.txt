configure_file(config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/config.hpp)

file(GLOB_RECURSE SPARSE_SOLVERS_HEADERS *.hpp ${CMAKE_CURRENT_BINARY_DIR}/config.hpp)
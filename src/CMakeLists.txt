foreach(PROG solve)
    file(GLOB ${PROG}_SOURCES ${PROG}*.cpp)
    
    add_executable(${PROG} ${${PROG}_SOURCES} ${SPARSE_SOLVERS_HEADERS})

    target_include_directories(${PROG} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${PROG} OpenMP::OpenMP_CXX Eigen3::Eigen)

endforeach()


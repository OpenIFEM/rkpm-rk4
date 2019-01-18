set(TARGET_SRC body.cpp
               boundary_conditions.cpp
               element.cpp
               gauss_face_integration.cpp
               gauss_integration.cpp
               kernel.cpp
               particle.cpp
               simulation_data.cpp
               simulation_time.cpp
               utilities.cpp)

add_library(rkpm-rk4 ${TARGET_SRC})
target_include_directories(rkpm-rk4 PRIVATE "${CMAKE_SOURCE_DIR}/inc")
target_include_directories(rkpm-rk4 PRIVATE "${EIGEN3_INCLUDE_DIR}")

file(GLOB public_headers "${CMAKE_SOURCE_DIR}/inc/*.h")
set_target_properties(rkpm-rk4 PROPERTIES PUBLIC_HEADER "${public_headers}")
install(TARGETS rkpm-rk4
        LIBRARY DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        PUBLIC_HEADER DESTINATION "${CMAKE_BINARY_DIR}/inc/rkpm-rk4")

set(tests bending
          bending-3D
          tensile)

foreach(test ${tests})
  set(output ${CMAKE_CURRENT_BINARY_DIR}/${test})
  file(MAKE_DIRECTORY ${output})
  add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}/${test}.cpp)
  target_include_directories(${test} PRIVATE "${CMAKE_SOURCE_DIR}/inc")
  target_include_directories(${test} PRIVATE "${EIGEN3_INCLUDE_DIR}")
  target_link_libraries(${test} rkpm-rk4 stdc++fs)
  add_test(NAME ${test} COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test} WORKING_DIRECTORY ${output})
endforeach()

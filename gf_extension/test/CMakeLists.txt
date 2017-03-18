include(ALPSEnableTests)

#set (test_src
    #ir_test
    #)
#
#if(ALPSCore_HAS_MPI)
    #foreach(test ${test_src})
        #alps_add_gtest(${test} NOMAIN PARTEST)
    #endforeach(test)
#else()
    #foreach(test ${test_src})
        #alps_add_gtest(${test})
    #endforeach(test)
#endif(ALPSCore_HAS_MPI)

#set(gf_tests ir_test)

#AD HOC SOLUTION
set (test_src ir_test)
foreach(test ${test_src})
    add_executable(ir_test ${test}.cpp)
    #message("DEBUG"  ${${PROJECT_NAME}_DEPENDS})
    target_link_libraries(${test} ${PROJECT_NAME} ${GTEST_MAIN_LIBRARIES}) #this works
    #target_link_libraries(${test} ${PROJECT_NAME} ${GTEST_MAIN_LIBRARIES} ${${PROJECT_NAME}_DEPENDS}) #this does not work
    set(cmd_ $<TARGET_FILE:${test}> ${test_xml_output_})
    #message("DEBUG"  ${GTEST_MAIN_LIBRARIES})
    add_test(NAME ${test} COMMAND ${cmd_})
endforeach()

include(ALPSEnableTests)

set (test_src ir_test )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

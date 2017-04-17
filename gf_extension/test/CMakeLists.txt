include(ALPSEnableTests)

set (test_src ir_basis gf_transform)

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

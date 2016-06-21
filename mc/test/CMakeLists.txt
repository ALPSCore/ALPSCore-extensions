include(ALPSEnableTests)

set (test_src
    mc_metropolis_test01
    mc_metropolis_test02
    )

if(ALPSCore_HAS_MPI)
    foreach(test ${test_src})
        alps_add_gtest(${test} NOMAIN PARTEST)
    endforeach(test)
else()
    foreach(test ${test_src})
        alps_add_gtest(${test})
    endforeach(test)
endif(ALPSCore_HAS_MPI)

include(ALPSEnableTests)

set (test_src
    mc_metropolis_test01
    mc_metropolis_test02
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

#add tests for MPI
if(ALPS_HAVE_MPI)

set (test_src_mpi
    mc_metropolis_test02
    )
foreach(test ${test_src_mpi})
    alps_add_gtest(${test} NOMAIN PARTEST)
endforeach(test)

endif(ALPS_HAVE_MPI)

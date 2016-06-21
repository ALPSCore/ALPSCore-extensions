#pragma once

#include <boost/local_function.hpp>
#include <boost/scoped_ptr.hpp>

#include "mpi_dispatcher.hpp"

namespace alps {

template<typename WrapType>
struct mpi_skel {
    mpi_skel(int njobs_max) { parts.reserve(njobs_max); }
    void add(WrapType &&x) { parts.push_back(std::forward<WrapType>(x)); };
    //void add(WrapType const& x){ parts.push_back(x); }
    std::map<int, int> run(const alps::mpi::communicator &comm, bool VerboseOutput = true);
    std::vector<WrapType> parts;
};

template<typename WrapType>
std::map<int, int> mpi_skel<WrapType>::run(const alps::mpi::communicator &comm, bool VerboseOutput) {
    int rank = comm.rank();
    int comm_size = comm.size();
    comm.barrier();
    if (rank == 0) { INFO("Calculating " << parts.size() << " jobs using " << comm_size << " procs."); };

    size_t ROOT = 0;
    boost::scoped_ptr<pMPI::MPIMaster> disp;

    if (comm.rank() == ROOT) {
        // prepare one Master on a root process for distributing parts.size() jobs
        std::vector<int> job_order(parts.size());
        for (size_t i = 0; i < job_order.size(); i++) job_order[i] = i;
//        for (size_t i=0; i<job_order.size(); i++) std::cout << job_order[i] << " " << std::flush; std::cout << std::endl; // DEBUG
//        for (size_t i=0; i<job_order.size(); i++) std::cout << parts[job_order[i]].complexity() << " " << std::flush; std::cout << std::endl; // DEBUG
        //std::sort(job_order.begin(), job_order.end(), [&](const int &l, const int &r){return (parts[l].complexity() > parts[r].complexity());});

        int BOOST_LOCAL_FUNCTION_TPL(bind this_, std::size_t l, std::size_t r) {
                return (this_->parts[l].complexity() > this_->parts[r].complexity());
            } BOOST_LOCAL_FUNCTION_NAME_TPL(comp1)
            std::sort(job_order.begin(), job_order.end(), comp1 );

            disp.reset(new pMPI::MPIMaster(comm, job_order, true));
        };

        comm.barrier();

        // Start calculating data
        for (pMPI::MPIWorker worker(comm, ROOT); !worker.is_finished();) {
            if (rank == ROOT) disp->order();
            worker.receive_order();
            //DEBUG((worker.Status == WorkerTag::Pending));
            if (worker.is_working()) { // for a specific worker
                int p = worker.current_job();
                if (VerboseOutput)
                    std::cout << "[" << p + 1 << "/" << parts.size() << "] P" << comm.rank()
                    << " : part " << p << " [" << parts[p].complexity() << "] run;" << std::endl;
                parts[p].run();
                worker.report_job_done();
            };
            if (rank == ROOT) disp->check_workers(); // check if there are free workers
        };
        // at this moment all communication is finished
        // Now spread the information, who did what.
        if (VerboseOutput && rank == ROOT) INFO("done.");
        comm.barrier();
        std::map<int, int> job_map;
        if (rank == ROOT) {
            job_map = disp->DispatchMap;
            std::vector<int> jobs(job_map.size());
            std::vector<int> workers(job_map.size());

            std::map<int, int>::const_iterator it = job_map.begin();
            for (int i = 0; i < workers.size(); i++) {
                jobs[i] = it->first;
                workers[i] = it->second;
                ++it;
            }
            //boost::mpi::broadcast(comm, jobs, ROOT);
            MPI_Bcast(&jobs, job_map.size(), MPI_INT, ROOT, comm); 
            //boost::mpi::broadcast(comm, workers, ROOT);
            MPI_Bcast(&workers, job_map.size(), MPI_INT, ROOT, comm);
        }
        else {
            std::vector<int> jobs(parts.size());
            //boost::mpi::broadcast(comm, jobs, ROOT);
            MPI_Bcast(&jobs, job_map.size(), MPI_INT, ROOT, comm); 
            std::vector<int> workers(parts.size());
            //boost::mpi::broadcast(comm, workers, ROOT);
            MPI_Bcast(&workers, job_map.size(), MPI_INT, ROOT, comm);
            for (size_t i = 0; i < jobs.size(); i++) job_map[jobs[i]] = workers[i];
        }
        return job_map;
    }

}; // end of namespace alps


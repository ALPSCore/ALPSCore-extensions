#pragma once

#include <alps/utilities/boost_mpi.hpp>
#include <stack>
#include <vector>
#include <map>

namespace alps {

namespace pMPI { 
    enum WorkerTag { Pending, Work, Finish }; // tags for MPI communication
}

struct MPIWorker {
    alps::mpi::communicator Comm;
    const int id;
    const int boss;

    MPIWorker(const alps::mpi::communicator &comm, int boss);
    void receive_order();
    void report_job_done();
    bool is_finished();
    bool is_working();


    int current_job() { return current_job_; };

protected:
    MPI_Request WorkReq, FinishReq;
    pMPI::WorkerTag Status;
    int current_job_;
    int finish_msg_;
    // temp messages go here
    int buf_;
};

struct MPIMaster {
    alps::mpi::communicator Comm;
    size_t Ntasks, Nprocs;

    std::stack<int> JobStack;
    std::stack<int> WorkerStack;

    std::map<int, int> DispatchMap;
    std::vector<int> task_numbers;

    std::vector<int> worker_pool;
    std::map<size_t, int> WorkerIndices;

    std::vector<MPI_Request> wait_statuses;
    std::vector<bool> workers_finish;

    MPIMaster(const alps::mpi::communicator &comm, std::vector<int> worker_pool, std::vector<int> task_numbers);
    MPIMaster(const alps::mpi::communicator &comm, std::vector<int> task_numbers, bool include_boss = true);
    MPIMaster(const alps::mpi::communicator &comm, size_t ntasks, bool include_boss = true);

    void swap(MPIMaster &x);
    void order_worker(int worker_id, int job);
    void order();
    void check_workers();
    void fill_stack_();
protected:
    // temp messages go here
    int buf_;
};

} // end of namespace alps



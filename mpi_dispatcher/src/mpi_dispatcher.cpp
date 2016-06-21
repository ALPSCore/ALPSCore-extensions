#include "alps/mpi_dispatcher/mpi_dispatcher.hpp"

namespace alps {

#ifndef NDEBUG
int MPI_DEBUG_VERBOSITY = 0; // set it to 1 to debug mpi messages
#endif

//
// Worker
//
MPIWorker::MPIWorker(const alps::mpi::communicator &comm, int boss) :
    Comm(comm),
    id(Comm.rank()),
    boss(boss),
    Status(pMPI::Pending),
    current_job_(-1) 
{ 
    MPI_Irecv(&current_job_, 1, MPI_INT, boss, int(pMPI::Work), Comm, &WorkReq); 
    MPI_Irecv(&finish_msg_, 1, MPI_INT, boss, int(pMPI::Finish), Comm, &FinishReq);
    //WorkReq(Comm.irecv(boss, int(pMPI::Work), current_job_)),
    //FinishReq(Comm.irecv(boss, int(pMPI::Finish))),

};


bool MPIWorker::is_finished() {
    return (Status == pMPI::Finish);
}

bool MPIWorker::is_working() {
    return (Status == pMPI::Work);
}

void MPIWorker::receive_order() {
    int work_test = 0;
    MPI_Test(&WorkReq, &work_test, MPI_STATUS_IGNORE);
    if (Status == pMPI::Pending && work_test) {
        Status = pMPI::Work;
        return;
    };
    int finish_test = 0;
    MPI_Test(&FinishReq, &finish_test, MPI_STATUS_IGNORE);
    if (Status == pMPI::Pending && finish_test) {
        Status = pMPI::Finish;
        //WorkReq.cancel();
        MPI_Cancel(&WorkReq);
        return;
    };
}

void MPIWorker::report_job_done() {
    //MPI_Request send_req = Comm.isend(boss, int(pMPI::Pending));
    MPI_Request send_req;
    MPI_Isend(&buf_, 1, MPI_INT, boss, int(pMPI::Pending), Comm, &send_req); 
    //DEBUG(id << "->" << boss << " tag: pending",MPI_DEBUG_VERBOSITY,1);
    Status = pMPI::Pending;
    //WorkReq = Comm.irecv(boss, int(pMPI::Work), current_job_);
    MPI_Irecv(&current_job_, 1, MPI_INT, boss, int(pMPI::Work), Comm, &WorkReq); 
};
//
// Master
//

void MPIMaster::fill_stack_() {
    for (int i = Ntasks - 1; i >= 0; i--) { JobStack.push(task_numbers[i]); };
    for (int p = Nprocs - 1; p >= 0; p--) {
        WorkerIndices[worker_pool[p]] = p;
        WorkerStack.push(worker_pool[p]);
    };
}


void MPIMaster::swap(MPIMaster &x) {
    std::swap(Ntasks, x.Ntasks);
    std::swap(Nprocs, x.Nprocs);
    std::swap(JobStack, x.JobStack);
    std::swap(WorkerStack, x.WorkerStack);
    std::swap(DispatchMap, x.DispatchMap);
    std::swap(task_numbers, x.task_numbers);
    std::swap(worker_pool, x.worker_pool);
    std::swap(WorkerIndices, x.WorkerIndices);
    std::swap(wait_statuses, x.wait_statuses);
    std::swap(workers_finish, x.workers_finish);
}

inline std::vector<int> _autorange_workers(const alps::mpi::communicator &comm, bool include_boss) {
    std::vector<int> out;
    size_t Nprocs(comm.size() - !include_boss);
    if (!Nprocs) throw (std::logic_error("No workers to evaluate"));
    for (size_t p = 0; p < comm.size(); p++) {
        if (include_boss || comm.rank() != p) {
            out.push_back(p);
        };
    };
    return out;
}

inline std::vector<int> _autorange_tasks(size_t ntasks) {
    std::vector<int> out(ntasks);
    for (size_t i = 0; i < ntasks; i++) {
        out[i] = i;
    };
    return out;
}

MPIMaster::MPIMaster(const alps::mpi::communicator &comm, std::vector<int> worker_pool, std::vector<int> task_numbers) :
    Comm(comm), Ntasks(task_numbers.size()),
    Nprocs(worker_pool.size()),
    task_numbers(task_numbers), worker_pool(worker_pool),
    wait_statuses(Nprocs),
    workers_finish(Nprocs, false) {
    fill_stack_();
};

MPIMaster::MPIMaster(const alps::mpi::communicator &comm, size_t ntasks, bool include_boss) :
    Comm(comm) {
    MPIMaster x(comm, _autorange_workers(comm, include_boss), _autorange_tasks(ntasks));
    this->swap(x);
};

MPIMaster::MPIMaster(const alps::mpi::communicator &comm, std::vector<int> task_numbers, bool include_boss) :
    Comm(comm) {
    MPIMaster x(comm, _autorange_workers(comm, include_boss), task_numbers);
    this->swap(x);
};

void MPIMaster::order_worker(int worker, int job) {
    //MPI_Request send_req = Comm.isend(worker, int(pMPI::Work), job);
    MPI_Request send_req;
    MPI_Isend(&job, 1, MPI_INT, worker, int(pMPI::Work), Comm, &send_req);
    //DEBUG(id << "->" << worker << " tag: work",MPI_DEBUG_VERBOSITY,1);
    //send_req.wait();
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
    DispatchMap[job] = worker;
    //wait_statuses[WorkerIndices[worker]] = Comm.irecv(worker, int(pMPI::Pending));
    MPI_Irecv(&buf_, 1, MPI_INT, worker, int(pMPI::Pending), Comm, &wait_statuses[WorkerIndices[worker]]); 
};

void MPIMaster::order() {
    while (!WorkerStack.empty() && !JobStack.empty()) {
        int &worker = WorkerStack.top();
        int &job = JobStack.top();
        order_worker(worker, job);
        WorkerStack.pop();
        JobStack.pop();
    };
};

void MPIMaster::check_workers() {
    if (!JobStack.empty()) {
        for (size_t i = 0; i < Nprocs && !JobStack.empty(); i++) {
            int wait_status_test = 0;
            MPI_Test(&wait_statuses[i], &wait_status_test, MPI_STATUS_IGNORE);
            if (wait_status_test) {
                WorkerStack.push(worker_pool[i]);
            };
        };
    }
    else {
        MPI_Request tmp_req;
        for (size_t i = 0; i < Nprocs; i++) {
            if (!workers_finish[i]) {
                //DEBUG(id << "->" << worker_pool[i] << " tag: finish",MPI_DEBUG_VERBOSITY,1);
                //Comm.isend(worker_pool[i], int(pMPI::Finish));
                MPI_Isend(&buf_, 1, MPI_INT, worker_pool[i], int(pMPI::Finish), Comm, &tmp_req); 
                workers_finish[i] = true; // to prevent double sending of Finish command that could overlap with other communication
            };
        };
    };
}

} // end of namespace alps


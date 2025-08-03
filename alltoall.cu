#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <iostream>
// Simplified NCCL communicator structure
struct NCCLComm {
    int rank;
    int nranks;
    
    // InfiniBand context and resources
    struct ibv_context* ib_ctx;        // IB device context
    struct ibv_pd* pd;                 // Protection domain
    struct ibv_cq* send_cq;            // Send completion queue  
    struct ibv_cq* recv_cq;            // Receive completion queue
    struct ibv_qp** qps;               // Queue pairs (one per remote rank)
    
    // Memory regions for RDMA
    struct ibv_mr** send_mrs;          // Send memory regions
    struct ibv_mr** recv_mrs;          // Receive memory regions
    
    // GPU memory buffers (registered for GPUDirect RDMA)
    void** gpu_send_buffers;           // GPU send buffers
    void** gpu_recv_buffers;           // GPU recv buffers
    
    // Work requests and scatter-gather lists
    struct ibv_send_wr* send_wrs;      // Send work requests
    struct ibv_recv_wr* recv_wrs;      // Receive work requests
    struct ibv_sge* send_sges;         // Send scatter-gather entries
    struct ibv_sge* recv_sges;         // Receive scatter-gather entries
};
/*
 * Naive implementation of all-to-all communication using NCCL. 
 * Each rank sends directly to every other rank.
 * Not optimal but shows the basic concept
*/
nccl_Result_t alltoall_basic(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    NcclComm* comm,
    cudaStream_t stream
) {
    size_t type_size = sizeof(float); // real NCCL has a lookup table
    size_t chunk_size = count * type_size;

    char* send_ptr = (char*)sendbuff;
    char* recv_ptr = (char*)recvbuff;

    printf("Rank %d: Posting recceives for %d ranks\n", comm->rank, comm->nranks);
    // Phase 1: post all receives first to avoid deadlock
    for (int src_rank = 0; src_rank < comm->nRanks; src_rank++) {
        if (src_rank == comm->rank) continue; // skip self
        // set up scatter gather entry
        comm->recv_sqes[src_rank].addr = (uintptr_t)(recv_ptr + src_rank * chunk_size);
        comm->recv_sqes[src_rank].length = chunk_size;
        comm->recv_sqes[src_rank].lkey = comm->recv_mrs[src_rank]->lkey;
        // set up receive work request
        comm->recv_wrs[src_rank].wr_id = src_rank;
        comm->recv_wrs[src_rank].next = NULL;
        comm->recv_wrs[src_rank].sg_list = &comm->recv_sqes[src_rank];
        comm->recv_wrs[src_rank].num_sge = 1;

        // Post non blocking receives
        struct ibv_recv_wr* bad_wr;
        int ret = ibv_post_recv(comm->qps[src_rank], &comm->recv_wrs[src_rank], &bad_wr);
        if (ret) {
            printf("Rank %d: Failed to post receive for rank %d\n", comm->rank, src_rank);
            return ncclInternalError;
        }
        printf("Rank %d: Posted receive for rank %d\n", comm->rank, src_rank);

        char* recv_chunk = recv_ptr + src_rank * chunk_size;
    }
    // step 2: reduce local data to outbuffer, no network needed
    char* own_send = send_ptr + comm->rank * chunk_size;
    char* own_recv = recv_ptr + comm->rank * chunk_size;
    cudaMemcpyAsync(own_recv, own_send, chunk_size, cudaMemcpyDeviceToDevice, stream);
    printf("Rank %d: Copied own data to recv buffer\n", comm->rank);
    // step 3: post all sends (after receives are ready)
    for (int dest_rank = 0; dest_rank < comm->nranks; dest_rank++) {
        if (dest_rank == comm->rank) continue; // skip self
        // Set up send scatter-gather entry
        comm->send_sges[dest_rank].addr = (uintptr_t)comm->gpu_send_buffers[dest_rank];
        comm->send_sges[dest_rank].length = chunk_size;
        comm->send_sges[dest_rank].lkey = comm->send_mrs[dest_rank]->lkey; // Local key
        // Set up send work request
        comm->send_wrs[dest_rank].wr_id = dest_rank;        // ID for completion
        comm->send_wrs[dest_rank].next = NULL;              // Single WR
        comm->send_wrs[dest_rank].opcode = IBV_WR_SEND;     // Standard send operation
        comm->send_wrs[dest_rank].send_flags = IBV_SEND_SIGNALED; // Generate completion
        comm->send_wrs[dest_rank].sg_list = &comm->send_sges[dest_rank];
        comm->send_wrs[dest_rank].num_sge = 1;              // One SGE
        // post non blocking send
        struct ibv_send_wr* bad_wr;
        int ret = ibv_post_send(comm->qps[dest_rank], &comm->sen d_wrs[dest_rank], &bad_wr);
        if (ret) {
            printf("Rank %d: Failed to post send for rank %d\n", comm->rank, dest_rank);
            return ncclInternalError;
        }
        printf("Rank %d: Posted send for rank %d\n", comm->rank, dest_rank);
    }
    // step 4: poll for completions
    printf("Rank %d: Polling for completions\n", comm->rank);
    int sends_completed = 0;
    int recvs_completed = 0;
    while ((sends_completed + recvs_completed) < expected_ops) {
        struct ibv_wc wcs[16]; //work completions
        int num_wc;
        // poll send completions
        num_wc = ibv_poll_cq(comm->send_cq, 16, wcs);
        for (int i = 0; i < num_wc; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                printf("Rank %d: Send completion error: %s\n", comm->rank, ibv_wc_status_str(wcs[i].status));
                return ncclInternalError;
            } else {
                sends_completed++;
                printf("Rank %d: Send completed for rank %d\n", comm->rank, wcs[i].wr_id);
            }
        }
        // poll receive completions
        num_wc = ibv_poll_cq(comm->recv_cq, 16, wcs);
        for (int i = 0; i < num_wc; i++) {
            if (wcs[i].status != IBV_WC_SUCCESS) {
                printf("Rank %d: Receive completion error: %s\n", comm->rank, ibv_wc_status_str(wcs[i].status));
                return ncclInternalError;
            } else {
                recvs_completed++;
                printf("Rank %d: Receive completed for rank %d\n", comm->rank, wcs[i].wr_id);
            }
        }
    }
    printf("Rank %d: All-to-all communication completed successfully\n", comm->rank);
    return ncclSuccess;
}
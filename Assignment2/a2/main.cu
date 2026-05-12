#include <cuda_runtime.h>
#include <omp.h>
#include<assert.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <limits.h>
#include "helper.h"
#include<random>
#include<set>
#include<string>
using namespace std;
std::random_device rd;
std::mt19937 gen(rd());


__global__ void knn(int* x,
                    int* y,
                    int* z,
                    int* I,
                    int* histogram,
                    int* cdf,
                    int* c_min,
                    int* updated_ints,
                    int n,
                    int k)
{
    __shared__ int y_tile_x[TILE];
    __shared__ int y_tile_y[TILE];
    __shared__ int y_tile_z[TILE];
    __shared__ int y_tile_I[TILE];

    int x_offset = blockIdx.x * TILE;
    int tid = threadIdx.x;
    int i = x_offset + tid;

    
    int pt_i_x,pt_i_y,pt_i_z,pt_i_I;
    if(i >= n){
        pt_i_I = 256;
    } else {
        
        pt_i_x = x[i];
        pt_i_y = y[i];
        pt_i_z = z[i];
        pt_i_I = I[i];
    }

    int dist_buf[MAX_PQ_CAPACITY];
    int idx_buf[MAX_PQ_CAPACITY];
    PriorityQueue pq;
    pq_init(&pq, dist_buf, idx_buf, MAX_PQ_CAPACITY);
    

    for(int t = 0; t < n; t += TILE){

        if(t + tid < n){
            y_tile_x[tid] = x[t+tid];
            y_tile_y[tid] = y[t+tid];
            y_tile_z[tid] = z[t+tid];
            y_tile_I[tid] = I[t+tid];
        }
            

        __syncthreads();

        if(pt_i_I != 256){

            int end = min(t + TILE, n);

            for(int j = t; j < end; j++){

                if(i == j) continue;

                
                int pt_j = j-t;
                int dist2 = dist_sq(pt_i_x,pt_i_y,pt_i_z,y_tile_x[pt_j],y_tile_y[pt_j],y_tile_z[pt_j]);
                if(pq.size == k){
                    int dist1,max_ind;
                    pq_peek(&pq,&dist1,&max_ind);
                    if(dist2 > dist1) continue;
                    else if(dist2 == dist1 && comp(x[max_ind],y[max_ind],z[max_ind],I[max_ind],y_tile_x[pt_j],y_tile_y[pt_j],y_tile_z[pt_j],y_tile_I[pt_j])){
                        continue;
                    }
                    
                    int _a,_b;
                    pq_extract_max(&pq,&_a,&_b,x,y,z,I);
                }

                
                pq_insert(&pq,dist2,j,x,y,z,I);
            }
        }

        __syncthreads();
    }

    if(i >= n || pt_i_I == 256) return;

    #define HIST(i,v) histogram[(i)*256 + (v)]
    #define CDF(i,v)  cdf[(i)*256 + (v)]

    
    

    HIST(i,pt_i_I)++;
    for(int idx = 0; idx < pq.size; idx++){
        
        int ind = idx_buf[idx];
        
        HIST(i, I[ind])++;
    }

    int running = 0;
    for(int v = 0; v < 256; v++){
        running += HIST(i, v);
        CDF(i, v) = running;
    }

    int cmin_val = 0;
    for(int v = 0; v < 256; v++){
        if(CDF(i, v) > 0){
            cmin_val = CDF(i, v);
            break;
        }
    }
    c_min[i] = cmin_val;

    int Ii = pt_i_I;
    int Ci = CDF(i, Ii);

    if(k+1 == cmin_val){
        updated_ints[i] = Ii;
    } else {
        int numerator = (Ci-cmin_val) * 255;
        int denominator = (k+1 - cmin_val);
        int val = numerator / denominator;
        updated_ints[i] = val;
    }

    #undef HIST
    #undef CDF
}


__global__ void assign_clusters(
    int* __restrict__ x,
    int* __restrict__ y,
    int* __restrict__ z,
    int* __restrict__ c_x,
    int* __restrict__ c_y,
    int* __restrict__ c_z,
    int* __restrict__ labels,
    int n,
    int k,
    int* d_changed
) {
    extern __shared__ int s_mem[];

    int* s_cx = s_mem;
    int* s_cy = s_mem + k;
    int* s_cz = s_mem + 2*k;

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + tid;

    
    for (int c = tid; c < k; c += blockDim.x) {
        s_cx[c] = c_x[c];
        s_cy[c] = c_y[c];
        s_cz[c] = c_z[c];
    }
    __syncthreads();

    if (i < n) {
        int px = x[i];
        int py = y[i];
        int pz = z[i];

        unsigned long long min_dist_sq = ULLONG_MAX;
        int best_cluster = -1;

        for (int c = 0; c < k; c++) {
            long long dx = (long long)px - s_cx[c];
            long long dy = (long long)py - s_cy[c];
            long long dz = (long long)pz - s_cz[c];

            unsigned long long dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_cluster = c;
            }
            else if (dist_sq == min_dist_sq) {
                
                if (best_cluster == -1 ||
                    (s_cx[c] > s_cx[best_cluster]) ||
                    (s_cx[c] == s_cx[best_cluster] && s_cy[c] > s_cy[best_cluster]) ||
                    (s_cx[c] == s_cx[best_cluster] && s_cy[c] == s_cy[best_cluster] && s_cz[c] > s_cz[best_cluster])) {
                    best_cluster = c;
                }
            }
        }

        if (labels[i] != best_cluster) {
            labels[i] = best_cluster;
            *d_changed = 1;
        }
    }
}
void k_means(
    int* x, int* y, int* z, int* I,
    int* d_x, int* d_y, int* d_z,
    int* updated_ints,
    int n, int k, int T
){
    int h_changed = 1;
    int iter = 0;

    int c_x[MAX_K], c_y[MAX_K], c_z[MAX_K];
    int h_cluster_sizes[MAX_K] = {0};

    int* labels = new int[n];

    int* d_km_buf;
    CHECK(cudaMalloc(&d_km_buf, (3 * k + n + 1) * sizeof(int)));
    int* d_cx = d_km_buf;
    int* d_cy = d_km_buf + k;
    int* d_cz = d_km_buf + 2 * k;
    int* d_labels = d_km_buf + 3 * k;
    int* d_changed = d_km_buf + 3 * k + n;

    
    for(int i = 0; i < k; i++){
        c_x[i] = x[i];
        c_y[i] = y[i];
        c_z[i] = z[i];
    }

    for(int i = 0; i < n; i++){
        labels[i] = -1;
    }

    CHECK(cudaMemcpy(d_labels, labels, n*sizeof(int), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (n + blockDim - 1) / blockDim;
    size_t shared_mem_size = 3 * k * sizeof(int);

    while(h_changed && iter < T){
        h_changed = 0;

        CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_cx, c_x, k*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_cy, c_y, k*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_cz, c_z, k*sizeof(int), cudaMemcpyHostToDevice));

        assign_clusters<<<gridDim, blockDim, shared_mem_size>>>(
            d_x, d_y, d_z,
            d_cx, d_cy, d_cz,
            d_labels, n, k, d_changed
        );
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(labels, d_labels, n*sizeof(int), cudaMemcpyDeviceToHost));

        
        long long sum_x[MAX_K] = {0};
        long long sum_y[MAX_K] = {0};
        long long sum_z[MAX_K] = {0};
        for(int c = 0; c < k; c++) h_cluster_sizes[c] = 0;

        #pragma omp parallel
        {
            long long loc_sum_x[MAX_K] = {0};
            long long loc_sum_y[MAX_K] = {0};
            long long loc_sum_z[MAX_K] = {0};
            int loc_counts[MAX_K] = {0};

            #pragma omp for
            for(int i = 0; i < n; i++){
                int cluster = labels[i];
                loc_sum_x[cluster] += x[i];
                loc_sum_y[cluster] += y[i];
                loc_sum_z[cluster] += z[i];
                loc_counts[cluster]++;
            }

            #pragma omp critical
            {
                for(int c = 0; c < k; c++){
                    sum_x[c] += loc_sum_x[c];
                    sum_y[c] += loc_sum_y[c];
                    sum_z[c] += loc_sum_z[c];
                    h_cluster_sizes[c] += loc_counts[c];
                }
            }
        }

        for(int c = 0; c < k; c++){
            if(h_cluster_sizes[c] > 0){
                c_x[c] = sum_x[c] / h_cluster_sizes[c];
                c_y[c] = sum_y[c] / h_cluster_sizes[c];
                c_z[c] = sum_z[c] / h_cluster_sizes[c];
            }
        }

        iter++;
    }

    

    int* h_hist_buf = (int*)calloc(2 * k * 256, sizeof(int));
    int* histogram = h_hist_buf;
    int* cdf       = h_hist_buf + k * 256;
    int c_min[MAX_K] = {0};

    for(int i = 0; i < n; i++){
        histogram[labels[i]*256 + I[i]]++;
    }

    #pragma omp parallel
    {
        #pragma omp for
        for(int kk = 0; kk < k; kk++){
            cdf[kk*256] = histogram[kk*256];
            c_min[kk] = histogram[kk*256];

            for(int i = 1; i < 256; i++){
                cdf[kk*256 + i] = cdf[kk*256 + i-1] + histogram[kk*256+i];
                if(!c_min[kk]) c_min[kk] = cdf[kk*256+i];
            }
        }

        #pragma omp for
        for(int i = 0; i < n; i++){
            int c = labels[i];
            if(c_min[c] == h_cluster_sizes[c]) {
                updated_ints[i] = I[i];
            } else {
                int numerator = (cdf[c*256 + I[i]] - c_min[c]) * 255;
                int denominator = h_cluster_sizes[c] - c_min[c];
                updated_ints[i] = numerator / denominator;
            }
        }
    }

    

    cudaFree(d_km_buf);

    delete[] labels;
    free(h_hist_buf);
}









__global__ void ivf_count_clusters(
    const int* __restrict__ labels,
    int* __restrict__ counts,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[labels[i]], 1);
}


__global__ void ivf_fill_lists(
    const int* __restrict__ labels,
    int* __restrict__ write_ptr,   
    int* __restrict__ list_points, 
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c   = labels[i];
        int pos = atomicAdd(&write_ptr[c], 1);
        list_points[pos] = i;
    }
}




__global__ void ivf_query_kernel(
    int* __restrict__ x,
    int* __restrict__ y,
    int* __restrict__ z,
    int* __restrict__ I,
    int* __restrict__ c_x,
    int* __restrict__ c_y,
    int* __restrict__ c_z,
    const int* __restrict__ list_points,
    const int* __restrict__ offsets,
    int* __restrict__ histogram,
    int* __restrict__ cdf,
    int* __restrict__ c_min,
    int* __restrict__ updated_ints,
    int n, int k, int nc, int n_probe
) {
    
    extern __shared__ int s_mem[];
    int* s_cx = s_mem;
    int* s_cy = s_mem + nc;
    int* s_cz = s_mem + 2 * nc;

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    for (int c = tid; c < nc; c += blockDim.x) {
        s_cx[c] = c_x[c];
        s_cy[c] = c_y[c];
        s_cz[c] = c_z[c];
    }
    __syncthreads();

    if (i >= n) return;

    int px = x[i], py = y[i], pz = z[i];

    
    unsigned long long cent_dist[MAX_K];
    for (int c = 0; c < nc; c++) {
        long long dx = (long long)px - s_cx[c];
        long long dy = (long long)py - s_cy[c];
        long long dz = (long long)pz - s_cz[c];
        cent_dist[c] = (unsigned long long)(dx*dx + dy*dy + dz*dz);
    }

    
    int dist_buf[MAX_PQ_CAPACITY];
    int idx_buf[MAX_PQ_CAPACITY];
    PriorityQueue pq;
    pq_init(&pq, dist_buf, idx_buf, k);

    
    for (int p = 0; p < n_probe; p++) {
        unsigned long long best_d = ULLONG_MAX;
        int best_c = 0;
        for (int c = 0; c < nc; c++) {
            if (cent_dist[c] < best_d) { best_d = cent_dist[c]; best_c = c; }
        }
        cent_dist[best_c] = ULLONG_MAX;

        int start = offsets[best_c];
        int end   = offsets[best_c + 1];

        for (int j = start; j < end; j++) {
            int idx = list_points[j];
            if (idx == i) continue;

            int d = dist_sq(px, py, pz, x[idx], y[idx], z[idx]);

            if (pq.size == k) {
                int max_d, max_idx;
                pq_peek(&pq, &max_d, &max_idx);
                if (d > max_d) continue;
                if (d == max_d && comp(x[max_idx], y[max_idx], z[max_idx], I[max_idx],
                                       x[idx],     y[idx],     z[idx],     I[idx]))
                    continue;
                pq_extract_max(&pq, &max_d, &max_idx, x, y, z, I);
            }
            pq_insert(&pq, d, idx, x, y, z, I);
        }
    }

    
    #define HIST(ii, v) histogram[(ii) * 256 + (v)]
    #define CDF(ii, v)  cdf[(ii) * 256 + (v)]

    int Ii = I[i];
    HIST(i, Ii)++;
    for (int t = 0; t < pq.size; t++) {
        HIST(i, I[idx_buf[t]])++;
    }

    int running = 0;
    for (int v = 0; v < 256; v++) {
        running += HIST(i, v);
        CDF(i, v) = running;
    }

    int cmin_val = 0;
    for (int v = 0; v < 256; v++) {
        if (CDF(i, v) > 0) { cmin_val = CDF(i, v); break; }
    }
    c_min[i] = cmin_val;

    int Ci = CDF(i, Ii);
    if (k + 1 == cmin_val) {
        updated_ints[i] = Ii;
    } else {
        updated_ints[i] = (Ci - cmin_val) * 255 / (k + 1 - cmin_val);
    }

    #undef HIST
    #undef CDF
}

void ivf_full(
    int* x, int* y, int* z, int* I,       
    int* d_x, int* d_y, int* d_z, int* d_I, 
    int n, int k, int T,
    int* d_histogram,    
    int* d_cdf,          
    int* d_c_min,        
    int* d_updated_ints  
) {

    int nc = n / k;
    int n_probe = (int)ceil(k * k / (double)n);
    if(n<=10000 || k <=10){
        nc = (int)ceil(sqrt(n));
        n_probe = nc;
    }
    


    if (nc < 1)     nc = 1;
    if (nc > MAX_K) nc = MAX_K;
    if (n_probe < 8)  n_probe = 8;
    if (n_probe > nc) n_probe = nc;
    

    int c_x[MAX_K], c_y[MAX_K], c_z[MAX_K];
    int h_cluster_sizes[MAX_K] = {0};
    int h_counts[MAX_K];
    int h_offsets[MAX_K + 1];

    int* labels = new int[n];

    int* d_ivf_buf;
    CHECK(cudaMalloc(&d_ivf_buf, (3 * nc + n + 1) * sizeof(int)));
    int* d_cx = d_ivf_buf;
    int* d_cy = d_ivf_buf + nc;
    int* d_cz = d_ivf_buf + 2 * nc;
    int* d_labels = d_ivf_buf + 3 * nc;
    int* d_changed = d_ivf_buf + 3 * nc + n;

    
    for (int i = 0; i < nc; i++) { c_x[i] = x[i]; c_y[i] = y[i]; c_z[i] = z[i]; }
    for (int i = 0; i < n; i++) labels[i] = -1;
    CHECK(cudaMemcpy(d_labels, labels, n * sizeof(int), cudaMemcpyHostToDevice));

    int blockSz = 256;
    int gridSz  = (n + blockSz - 1) / blockSz;
    size_t shmem = 3 * nc * sizeof(int);

    int h_changed = 1, iter = 0;
    while (h_changed && iter < T) {
        h_changed = 0;
        CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int),     cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_cx,      c_x,        nc * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_cy,      c_y,        nc * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_cz,      c_z,        nc * sizeof(int), cudaMemcpyHostToDevice));

        assign_clusters<<<gridSz, blockSz, shmem>>>(
            d_x, d_y, d_z, d_cx, d_cy, d_cz, d_labels, n, nc, d_changed);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),    cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(labels,     d_labels,  n * sizeof(int), cudaMemcpyDeviceToHost));

        
        long long sum_x[MAX_K] = {0};
        long long sum_y[MAX_K] = {0};
        long long sum_z[MAX_K] = {0};
        for (int c = 0; c < nc; c++) h_cluster_sizes[c] = 0;

        #pragma omp parallel
        {
            long long lx[MAX_K] = {0};
            long long ly[MAX_K] = {0};
            long long lz[MAX_K] = {0};
            int       lc[MAX_K] = {0};

            #pragma omp for
            for (int ii = 0; ii < n; ii++) {
                int cl = labels[ii];
                lx[cl] += x[ii]; ly[cl] += y[ii]; lz[cl] += z[ii]; lc[cl]++;
            }
            #pragma omp critical
            {
                for (int c = 0; c < nc; c++) {
                    sum_x[c] += lx[c]; sum_y[c] += ly[c]; sum_z[c] += lz[c];
                    h_cluster_sizes[c] += lc[c];
                }
            }
        }

        for (int c = 0; c < nc; c++) {
            if (h_cluster_sizes[c] > 0) {
                c_x[c] = (int)(sum_x[c] / h_cluster_sizes[c]);
                c_y[c] = (int)(sum_y[c] / h_cluster_sizes[c]);
                c_z[c] = (int)(sum_z[c] / h_cluster_sizes[c]);
            }
        }
        iter++;
    }

    
    CHECK(cudaMemcpy(d_cx, c_x, nc * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cy, c_y, nc * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cz, c_z, nc * sizeof(int), cudaMemcpyHostToDevice));

    
    int* d_ivf2_buf;
    CHECK(cudaMalloc(&d_ivf2_buf, (3 * nc + 1 + n) * sizeof(int)));
    int* d_counts = d_ivf2_buf;
    int* d_offsets = d_ivf2_buf + nc;
    int* d_write_ptr = d_ivf2_buf + 2 * nc + 1;
    int* d_list_points = d_ivf2_buf + 3 * nc + 1;
    CHECK(cudaMemset(d_counts, 0, nc * sizeof(int)));

    ivf_count_clusters<<<gridSz, blockSz>>>(d_labels, d_counts, n);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_counts, d_counts, nc * sizeof(int), cudaMemcpyDeviceToHost));

    h_offsets[0] = 0;
    for (int c = 0; c < nc; c++) h_offsets[c + 1] = h_offsets[c] + h_counts[c];

    CHECK(cudaMemcpy(d_offsets, h_offsets, (nc + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_write_ptr, d_offsets, nc * sizeof(int), cudaMemcpyDeviceToDevice));
    ivf_fill_lists<<<gridSz, blockSz>>>(d_labels, d_write_ptr, d_list_points, n);
    CHECK(cudaDeviceSynchronize());

    
    ivf_query_kernel<<<gridSz, blockSz, shmem>>>(
        d_x, d_y, d_z, d_I,
        d_cx, d_cy, d_cz,
        d_list_points, d_offsets,
        d_histogram, d_cdf, d_c_min, d_updated_ints,
        n, k, nc, n_probe
    );
    CHECK(cudaDeviceSynchronize());

    
    cudaFree(d_ivf_buf);
    cudaFree(d_ivf2_buf);

    delete[] labels;
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <knn|kmeans|approx_knn>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = argv[2];

    bool run_knn = false;
    bool run_kmeans = false;
    bool run_approx = false;

    if (mode == "knn") {
        run_knn = true;
    }
    else if (mode == "kmeans") {
        run_kmeans = true;
    }
    else if (mode == "approx_knn") {
        run_approx = true;
    }
    else {
        std::cerr << "Invalid mode: " << mode << "\n";
        return 1;
    }


    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    int n, k, T;
    in >> n >> k >> T;

    int* h_buf = new int[5 * n];
    int* x = h_buf;
    int* y = h_buf + n;
    int* z = h_buf + 2 * n;
    int* I = h_buf + 3 * n;
    int* updated_ints = h_buf + 4 * n;

    for(int i = 0; i < n; i++){
        in >> x[i] >> y[i] >> z[i] >> I[i];
    }
    in.close();

    int* d_buf;
    CHECK(cudaMalloc(&d_buf, 5 * (size_t)n * sizeof(int)));
    int* d_x = d_buf;
    int* d_y = d_buf + n;
    int* d_z = d_buf + 2 * n;
    int* d_I = d_buf + 3 * n;
    int* d_updated_ints = d_buf + 4 * n;
    CHECK(cudaMemcpy(d_buf, h_buf, 4 * (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    if (run_knn) {
        int* d_hist_buf;
        CHECK(cudaMalloc(&d_hist_buf, (2 * (size_t)n * 256 + n) * sizeof(int)));
        int* d_histogram = d_hist_buf;
        int* d_cdf = d_hist_buf + (size_t)n * 256;
        int* c_min = d_hist_buf + 2 * (size_t)n * 256;

        CHECK(cudaMemset(d_histogram, 0, n * 256 * sizeof(int)));

        int blockDim = TILE;
        int gridDim = (n + TILE - 1) / TILE;
        
        knn<<<gridDim, blockDim>>>(d_x,d_y,d_z,d_I, d_histogram, d_cdf, c_min, d_updated_ints, n, k);
        cudaDeviceSynchronize();

        CHECK(cudaMemcpy(updated_ints, d_updated_ints, n * sizeof(int), cudaMemcpyDeviceToHost));

        std::ofstream out_knn("knn.txt");
        for(int i = 0; i < n; i++){
            out_knn << x[i] << " "
                    << y[i] << " "
                    << z[i] << " "
                    << updated_ints[i] << "\n";
        }
        out_knn.close();

        cudaFree(d_hist_buf);
    }

    if (run_kmeans) {
        k_means(x,y,z,I,d_x,d_y,d_z, updated_ints, n, k, T);

        std::ofstream out_kmeans("kmeans.txt");
        for(int i = 0; i < n; i++){
            out_kmeans << x[i] << " "
                       << y[i] << " "
                       << z[i] << " "
                       << updated_ints[i] << "\n";
        }
        out_kmeans.close();
    }
    if(run_approx){
        int* d_hist_buf;
        CHECK(cudaMalloc(&d_hist_buf, (2 * (size_t)n * 256 + n) * sizeof(int)));
        int* d_histogram = d_hist_buf;
        int* d_cdf = d_hist_buf + (size_t)n * 256;
        int* c_min = d_hist_buf + 2 * (size_t)n * 256;

        CHECK(cudaMemset(d_histogram, 0, n * 256 * sizeof(int)));
        
        if(n<10000 || k <= 32){
            int blockDim = TILE;
            int gridDim = (n + TILE - 1) / TILE;
            
            knn<<<gridDim, blockDim>>>(d_x,d_y,d_z,d_I, d_histogram, d_cdf, c_min, d_updated_ints, n, k);
            cudaDeviceSynchronize();
        }


        else ivf_full(x,y,z,I,d_x,d_y,d_z,d_I,n,k,T,d_histogram, d_cdf, c_min, d_updated_ints);
        CHECK(cudaMemcpy(updated_ints, d_updated_ints, n*sizeof(int), cudaMemcpyDeviceToHost));
        std::ofstream out_a_knn("approx_knn.txt");
        for(int i = 0; i < n; i++){
            out_a_knn << x[i] << " "
                      << y[i] << " "
                      << z[i] << " "
                      << updated_ints[i] << "\n";
        }
        out_a_knn.close();
        cudaFree(d_hist_buf);
    }


    cudaFree(d_buf);

    delete[] h_buf;
    
    return 0;
}

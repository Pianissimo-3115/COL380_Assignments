// #define unsigned long long
#ifndef CUDA_CHECK
#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}
#endif

#define K 12
#define B 12
#define NUM_PROBES 4
#define BUCKET_MASK ((1 << B) - 1)
#define MAX_K 128




#define TILE 128

#ifndef MAX_PQ_CAPACITY
#define MAX_PQ_CAPACITY TILE
#endif



#pragma once

struct PriorityQueue {
    int *key;
    int *val;
    int size;
    int capacity;
};

__device__ __forceinline__
void pq_init(PriorityQueue *pq, int *key_buf, int *val_buf, int capacity) {
    pq->key = key_buf;
    pq->val = val_buf;
    pq->size = 0;
    pq->capacity = capacity;
}

__device__ __forceinline__
void pq_swap(PriorityQueue *pq, int i, int j) {
    int tk = pq->key[i];
    pq->key[i] = pq->key[j];
    pq->key[j] = tk;

    int tv = pq->val[i];
    pq->val[i] = pq->val[j];
    pq->val[j] = tv;
}

__device__ __forceinline__
int better(PriorityQueue *pq, int i, int j,
           int *x, int *y, int *z, int *I) {

    int k1 = pq->key[i];
    int k2 = pq->key[j];

    if (k1 != k2) return k1 > k2;

    int v1 = pq->val[i];
    int v2 = pq->val[j];

    int x1 = x[v1], x2 = x[v2];
    if (x1 != x2) return x1 > x2;

    int y1 = y[v1], y2 = y[v2];
    if (y1 != y2) return y1 > y2;

    int z1 = z[v1], z2 = z[v2];
    if (z1 != z2) return z1 > z2;

    return I[v1] > I[v2];
}

__device__ __forceinline__
void heapify_up(PriorityQueue *pq, int i,
                int *x, int *y, int *z, int *I) {

    while (i > 0) {
        int p = (i - 1) >> 1;
        if (better(pq, p, i, x, y, z, I)) break;
        pq_swap(pq, p, i);
        i = p;
    }
}

__device__ __forceinline__
void heapify_down(PriorityQueue *pq, int i,
                  int *x, int *y, int *z, int *I) {

    int size = pq->size;

    while (1) {
        int l = (i << 1) + 1;
        if (l >= size) break;

        int r = l + 1;
        int largest = l;

        if (r < size && better(pq, r, l, x, y, z, I))
            largest = r;

        if (better(pq, i, largest, x, y, z, I)) break;

        pq_swap(pq, i, largest);
        i = largest;
    }
}

__device__ __forceinline__
void pq_insert(PriorityQueue *pq, int key, int val,
               int *x, int *y, int *z, int *I) {

    if (pq->size >= pq->capacity) return;

    int i = pq->size++;
    pq->key[i] = key;
    pq->val[i] = val;

    heapify_up(pq, i, x, y, z, I);
}

__device__ __forceinline__
int pq_extract_max(PriorityQueue *pq, int *key, int *val,
                   int *x, int *y, int *z, int *I) {

    if (pq->size == 0) return 0;

    *key = pq->key[0];
    *val = pq->val[0];

    int last = --pq->size;
    pq->key[0] = pq->key[last];
    pq->val[0] = pq->val[last];

    heapify_down(pq, 0, x, y, z, I);
    return 1;
}

__device__ __forceinline__
int pq_peek(PriorityQueue *pq, int *key, int *val) {

    if (pq->size == 0) return 0;

    *key = pq->key[0];
    *val = pq->val[0];
    return 1;
}

__host__ __device__ static inline int comp(int x1, int y1, int z1, int I1, int x2, int y2, int z2, int I2){
    if(x1 == x2){
        if(y1 == y2){
            if(z1 == z2) return I1 < I2;
            return z1 < z2;
        }
        return y1 < y2;
    }
    return x1 < x2;
}

// typedef struct {
//     int      x;
//     int      y;
//     int      z;
//     unsigned I;
// } point;
//
// typedef struct {
//     point origin;
//     point data[MAX_PQ_CAPACITY];
//     int   size;
// } PriorityQueue;
//
//
__host__ __device__ static inline int dist_sq(int x1, int y1, int z1, int x2, int y2, int z2)
{
    int dx = (x1 - x2);
    int dy = (y1 - y2);
    int dz = (z1 - z2);
    return dx*dx + dy*dy + dz*dz;
}

// __host__ __device__ static inline void _pq_swap(point* a, point* b)
// {
//     point tmp = *a;
//     *a = *b;
//     *b = tmp;
// }
//
//
// __host__ __device__ static inline void _pq_sift_up(PriorityQueue* pq, int i)
// {
//     while (i > 0) {
//         int parent = (i - 1) / 2;
//         if (_pq_dist_sq(pq->origin, pq->data[i]) >
//             _pq_dist_sq(pq->origin, pq->data[parent]))
//         {
//             _pq_swap(&pq->data[i], &pq->data[parent]);
//             i = parent;
//         } else {
//             break;
//         }
//     }
// }
//
// __host__ __device__ static inline void _pq_sift_down(PriorityQueue* pq, int i)
// {
//     while (1) {
//         int largest = i;
//         int left    = 2*i + 1;
//         int right   = 2*i + 2;
//
//         if (left  < pq->size &&
//             _pq_dist_sq(pq->origin, pq->data[left]) >
//             _pq_dist_sq(pq->origin, pq->data[largest]))
//             largest = left;
//
//         if (right < pq->size &&
//             _pq_dist_sq(pq->origin, pq->data[right]) >
//             _pq_dist_sq(pq->origin, pq->data[largest]))
//             largest = right;
//
//         if (largest == i) break;
//
//         _pq_swap(&pq->data[i], &pq->data[largest]);
//         i = largest;
//     }
// }
//
// __host__ __device__ static inline void pq_init(PriorityQueue* pq, point origin)
// {
//     pq->origin = origin;
//     pq->size   = 0;
// }
//
// __host__ __device__ static inline int pq_empty(const PriorityQueue* pq)
// {
//     return pq->size == 0;
// }
//
// __host__ __device__ static inline int pq_full(const PriorityQueue* pq)
// {
//     return pq->size == MAX_PQ_CAPACITY;
// }
//
// __host__ __device__ static inline void pq_push(PriorityQueue* pq, point p)
// {
//     if (pq->size == MAX_PQ_CAPACITY) return;   // no room
//     pq->data[pq->size++] = p;
//     _pq_sift_up(pq, pq->size - 1);
// }
//
// __host__ __device__ static inline point pq_top(const PriorityQueue* pq)
// {
//     return pq->data[0];
// }
//
// __host__ __device__ static inline point pq_pop(PriorityQueue* pq)
// {
//     point top        = pq->data[0];
//     pq->data[0]      = pq->data[--pq->size];
//     _pq_sift_down(pq, 0);
//     return top;
// }
//
// __host__ __device__ static inline long long pq_top_dist_sq(const PriorityQueue* pq)
// {
//     return _pq_dist_sq(pq->origin, pq->data[0]);
// }
//
// #endif

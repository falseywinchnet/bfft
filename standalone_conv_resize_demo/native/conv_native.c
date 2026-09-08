#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define CONV_NEON 1
#else
#define CONV_NEON 0
#endif

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

static const float CURRENT_FIR[5][6] = {
    {4.0f/240.0f, -32.0f/240.0f, 0.0f, 32.0f/240.0f, -4.0f/240.0f, 0.0f},
    {3.0f/240.0f, -16.0f/240.0f, -30.0f/240.0f, 48.0f/240.0f, -5.0f/240.0f, 0.0f},
    {-7.0f/240.0f, 39.0f/240.0f, -130.0f/240.0f, 130.0f/240.0f, -39.0f/240.0f, 7.0f/240.0f},
    {0.0f, 5.0f/240.0f, -48.0f/240.0f, 30.0f/240.0f, 16.0f/240.0f, -3.0f/240.0f},
    {0.0f, 4.0f/240.0f, -32.0f/240.0f, 0.0f, 32.0f/240.0f, -4.0f/240.0f}
};

API int conv_backend_version(void) { return 1; }
API int conv_backend_has_neon(void) { return CONV_NEON; }

typedef void (*parallel_function)(void *, int, int);

static int configured_threads(void) {
    const char *setting = getenv("CONV_NATIVE_THREADS");
    if (setting && *setting) {
        const int requested = atoi(setting);
        if (requested > 0) return requested > 64 ? 64 : requested;
    }
    const long detected = sysconf(_SC_NPROCESSORS_ONLN);
    if (detected < 1) return 1;
    return detected > 64 ? 64 : (int)detected;
}

API int conv_backend_threads(void) { return configured_threads(); }

typedef struct native_pool native_pool;
typedef struct { native_pool *pool; int index; } pool_argument;

struct native_pool {
    pthread_mutex_t state;
    pthread_mutex_t dispatch;
    pthread_cond_t start;
    pthread_cond_t done;
    pthread_t threads[63];
    pool_argument arguments[63];
    parallel_function function;
    void *context;
    int maximum_workers;
    int active_workers;
    int chunk;
    int count;
    int generation;
    int completed;
};

static native_pool GLOBAL_POOL;
static pthread_once_t POOL_ONCE=PTHREAD_ONCE_INIT;

static void *pool_worker(void *opaque){
    pool_argument *argument=(pool_argument *)opaque;
    native_pool *pool=argument->pool;
    const int index=argument->index;
    int seen=0;
    pthread_mutex_lock(&pool->state);
    for(;;){
        while(pool->generation==seen)pthread_cond_wait(&pool->start,&pool->state);
        seen=pool->generation;
        const int active=pool->active_workers;
        const int begin=index*pool->chunk;
        int end=begin+pool->chunk;if(end>pool->count)end=pool->count;
        parallel_function function=pool->function;void *context=pool->context;
        const int has_work=index<active&&begin<end;
        pthread_mutex_unlock(&pool->state);
        if(has_work)function(context,begin,end);
        pthread_mutex_lock(&pool->state);
        if(has_work&&++pool->completed==active-1)pthread_cond_signal(&pool->done);
    }
}

static void initialize_pool(void){
    native_pool *pool=&GLOBAL_POOL;
    memset(pool,0,sizeof(*pool));
    pthread_mutex_init(&pool->state,NULL);pthread_mutex_init(&pool->dispatch,NULL);
    pthread_cond_init(&pool->start,NULL);pthread_cond_init(&pool->done,NULL);
    pool->maximum_workers=configured_threads();
    for(int index=1;index<pool->maximum_workers;++index){
        pool->arguments[index-1].pool=pool;pool->arguments[index-1].index=index;
        if(pthread_create(&pool->threads[index-1],NULL,pool_worker,
                          &pool->arguments[index-1])!=0){pool->maximum_workers=index;break;}
    }
}

static void parallel_for(int count, int minimum_grain,
                         parallel_function function, void *context) {
    if (count <= 0) return;
    pthread_once(&POOL_ONCE,initialize_pool);
    native_pool *pool=&GLOBAL_POOL;
    int workers=pool->maximum_workers;
    const int useful = (count + minimum_grain - 1) / minimum_grain;
    if (workers > useful) workers = useful;
    if (workers <= 1) { function(context, 0, count); return; }
    pthread_mutex_lock(&pool->dispatch);
    pthread_mutex_lock(&pool->state);
    pool->function=function;pool->context=context;pool->count=count;
    pool->active_workers=workers;pool->chunk=(count+workers-1)/workers;
    pool->completed=0;++pool->generation;
    pthread_cond_broadcast(&pool->start);
    pthread_mutex_unlock(&pool->state);
    function(context,0,pool->chunk<count?pool->chunk:count);
    pthread_mutex_lock(&pool->state);
    while(pool->completed<workers-1)pthread_cond_wait(&pool->done,&pool->state);
    pthread_mutex_unlock(&pool->state);
    pthread_mutex_unlock(&pool->dispatch);
}

static inline float first_jet(const float *x, int n, int lanes, int i, int lane) {
    if (i == 0) return (-3.0f*x[lane] + 4.0f*x[lanes+lane] - x[2*lanes+lane]) * 0.5f;
    if (i == 1) return (x[2*lanes+lane] - x[lane]) * 0.5f;
    if (i == n-2) return (x[(n-1)*lanes+lane] - x[(n-3)*lanes+lane]) * 0.5f;
    if (i == n-1) return (3.0f*x[(n-1)*lanes+lane] - 4.0f*x[(n-2)*lanes+lane] + x[(n-3)*lanes+lane]) * 0.5f;
    return (x[(i-2)*lanes+lane] - 8.0f*x[(i-1)*lanes+lane]
            + 8.0f*x[(i+1)*lanes+lane] - x[(i+2)*lanes+lane]) / 12.0f;
}

static inline float second_jet(const float *x, int n, int lanes, int i, int lane) {
    if (i == 0) return 2.0f*x[lane] - 5.0f*x[lanes+lane] + 4.0f*x[2*lanes+lane] - x[3*lanes+lane];
    if (i == 1) return x[lane] - 2.0f*x[lanes+lane] + x[2*lanes+lane];
    if (i == n-2) return x[(n-3)*lanes+lane] - 2.0f*x[(n-2)*lanes+lane] + x[(n-1)*lanes+lane];
    if (i == n-1) return 2.0f*x[(n-1)*lanes+lane] - 5.0f*x[(n-2)*lanes+lane]
                         + 4.0f*x[(n-3)*lanes+lane] - x[(n-4)*lanes+lane];
    return (-x[(i+2)*lanes+lane] + 16.0f*x[(i+1)*lanes+lane]
            - 30.0f*x[i*lanes+lane] + 16.0f*x[(i-1)*lanes+lane]
            - x[(i-2)*lanes+lane]) / 12.0f;
}

typedef struct {
    const float *x;
    int n;
    int lanes;
    float *raw;
} raw_context;

static void raw_currents_worker(void *opaque, int begin, int end) {
    raw_context *job = (raw_context *)opaque;
    const float *x = job->x;
    const int n = job->n, lanes = job->lanes;
    float *raw = job->raw;
    for (int cell = begin; cell < end; ++cell) {
        int lane = 0;
        if (cell >= 2 && cell < n - 3) {
#if CONV_NEON
            for (; lane + 4 <= lanes; lane += 4) {
                float32x4_t acc[5] = {
                    vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0),
                    vdupq_n_f32(0), vdupq_n_f32(0)
                };
                for (int tap = 0; tap < 6; ++tap) {
                    float32x4_t value = vld1q_f32(x + (cell - 2 + tap)*lanes + lane);
                    for (int k = 0; k < 5; ++k)
                        acc[k] = vfmaq_n_f32(acc[k], value, CURRENT_FIR[k][tap]);
                }
                for (int k = 0; k < 5; ++k)
                    vst1q_f32(raw + (cell*5 + k)*lanes + lane, acc[k]);
            }
#endif
            for (; lane < lanes; ++lane) {
                for (int k = 0; k < 5; ++k) {
                    float sum = 0.0f;
                    for (int tap = 0; tap < 6; ++tap)
                        sum = fmaf(
                            CURRENT_FIR[k][tap],
                            x[(cell - 2 + tap)*lanes + lane], sum
                        );
                    raw[(cell*5+k)*lanes+lane] = sum;
                }
            }
        } else {
            for (; lane < lanes; ++lane) {
                const float left = x[cell*lanes+lane];
                const float right = x[(cell+1)*lanes+lane];
                const float delta = right - left;
                const float m0 = first_jet(x, n, lanes, cell, lane);
                const float m1 = first_jet(x, n, lanes, cell+1, lane);
                const float q0 = second_jet(x, n, lanes, cell, lane);
                const float q1 = second_jet(x, n, lanes, cell+1, lane);
                raw[(cell*5+0)*lanes+lane] = m0 / 5.0f;
                raw[(cell*5+1)*lanes+lane] = m0 / 5.0f + q0 / 20.0f;
                raw[(cell*5+2)*lanes+lane] = delta - 0.4f*(m0+m1) + (q1-q0)/20.0f;
                raw[(cell*5+3)*lanes+lane] = m1 / 5.0f - q1 / 20.0f;
                raw[(cell*5+4)*lanes+lane] = m1 / 5.0f;
            }
        }
    }
}

static void raw_currents(const float *x, int n, int lanes, float *raw) {
    raw_context context = {x,n,lanes,raw};
    parallel_for(n-1, 24, raw_currents_worker, &context);
}

static inline void compare_swap(float *a, float *b) {
    if (*a > *b) { const float t=*a; *a=*b; *b=t; }
}

static inline void sort_five(float value[5]) {
    /* Fixed nine-comparator network; avoids a function call and indirect
       comparator for every five-current fibre. */
    compare_swap(value+0,value+1); compare_swap(value+3,value+4);
    compare_swap(value+2,value+4); compare_swap(value+2,value+3);
    compare_swap(value+1,value+4); compare_swap(value+0,value+3);
    compare_swap(value+0,value+2); compare_swap(value+1,value+3);
    compare_swap(value+1,value+2);
}

static void simplex_five(const float a[5],float total,float c[5]){
    if(total<=0.0f){for(int k=0;k<5;++k)c[k]=0.0f;return;}
    float ordered[5];memcpy(ordered,a,sizeof(ordered));sort_five(ordered);
    double prefix=0.0,theta=0.0;
    for(int rank=1;rank<=5;++rank){
        const double value=ordered[5-rank];prefix+=value;
        const double candidate=(prefix-total)/rank;
        if(value>candidate)theta=candidate;
    }
    double mass=0.0;int best=0;
    for(int k=0;k<5;++k){c[k]=(float)fmax(0.0,(double)a[k]-theta);mass+=c[k];if(c[k]>c[best])best=k;}
    c[best]+=(float)((double)total-mass);
}

static void project_fibre(const float a[5], const int8_t sign[5], float delta, float c[5]) {
    int all_positive=1,all_negative=1;
    for(int k=0;k<5;++k){all_positive&=sign[k]>0;all_negative&=sign[k]<0;}
    if(all_positive&&delta>=0.0f){simplex_five(a,delta,c);return;}
    if(all_negative&&delta<=0.0f){
        float reflected[5],projected[5];
        for(int k=0;k<5;++k)reflected[k]=-a[k];
        simplex_five(reflected,-delta,projected);
        for(int k=0;k<5;++k)c[k]=-projected[k];
        return;
    }
    float breaks[5];
    memcpy(breaks, a, sizeof(breaks));
    sort_five(breaks);
    int found = 0;
    for (int region = 0; region <= 5 && !found; ++region) {
        const double low = region == 0 ? -INFINITY : breaks[region-1];
        const double high = region == 5 ? INFINITY : breaks[region];
        double probe;
        if (!isfinite(low)) probe = high - fmax(1.0, fabs(high));
        else if (!isfinite(high)) probe = low + fmax(1.0, fabs(low));
        else probe = 0.5 * (low + high);
        int count = 0;
        double sum = 0.0;
        for (int k = 0; k < 5; ++k) {
            const int active = sign[k] > 0 ? a[k] > probe : a[k] < probe;
            if (active) { ++count; sum += a[k]; }
        }
        if (!count) continue;
        const double lambda = (sum - delta) / count;
        const double tol = 2.0e-6 * fmax(1.0, fmax(fabs(low), fabs(high)));
        if (lambda < low - tol || lambda > high + tol) continue;
        double mass = 0.0;
        for (int k = 0; k < 5; ++k) {
            const double value = (double)a[k] - lambda;
            c[k] = sign[k] > 0 ? (float)fmax(0.0, value) : (float)fmin(0.0, value);
            mass += c[k];
        }
        if (fabs(mass - delta) <= 2.0e-5 * fmax(1.0, fabs(delta))) found = 1;
    }
    if (!found) {
        double span = 1.0 + fabs(delta);
        for (int k = 0; k < 5; ++k) span += fabs(a[k]);
        double lo = breaks[0] - span, hi = breaks[4] + span;
        for (int it = 0; it < 64; ++it) {
            const double lambda = 0.5 * (lo + hi);
            double mass = 0.0;
            for (int k = 0; k < 5; ++k) {
                const double value = a[k] - lambda;
                mass += sign[k] > 0 ? fmax(0.0, value) : fmin(0.0, value);
            }
            if (mass > delta) lo = lambda; else hi = lambda;
        }
        const double lambda = 0.5 * (lo + hi);
        for (int k = 0; k < 5; ++k) {
            const double value = a[k] - lambda;
            c[k] = sign[k] > 0 ? (float)fmax(0.0, value) : (float)fmin(0.0, value);
        }
    }
    double mass = 0.0;
    int best = -1;
    float magnitude = -1.0f;
    for (int k = 0; k < 5; ++k) {
        mass += c[k];
        if (fabsf(c[k]) > magnitude) { magnitude = fabsf(c[k]); best = k; }
    }
    if (best >= 0) c[best] += (float)((double)delta - mass);
}

typedef struct {
    const float *x;
    int n;
    int lanes;
    float *current;
} projection_context;

static void ordered_projection_worker(void *opaque, int lane_begin, int lane_end) {
    projection_context *job = (projection_context *)opaque;
    const float *x = job->x;
    const int n = job->n, lanes = job->lanes;
    float *current = job->current;
    const int cells = n-1, slots = cells*5;
    int8_t *coarse = (int8_t *)malloc((size_t)cells);
    int8_t *ledger = (int8_t *)malloc((size_t)slots);
    int *boundaries = (int *)malloc((size_t)cells * sizeof(int));
    int8_t *new_signs = (int8_t *)malloc((size_t)cells);
    if (!coarse || !ledger || !boundaries || !new_signs) abort();
    for (int lane = lane_begin; lane < lane_end; ++lane) {
        int any = 0;
        for (int cell = 0; cell < cells; ++cell) {
            const float d = x[(cell+1)*lanes+lane] - x[cell*lanes+lane];
            coarse[cell] = d > 0.0f ? 1 : d < 0.0f ? -1 : 0;
            any |= coarse[cell] != 0;
        }
        if (!any) {
            for (int slot = 0; slot < slots; ++slot) current[slot*lanes+lane] = 0.0f;
            continue;
        }
        for (int i = 1; i < cells; ++i) if (!coarse[i]) coarse[i] = coarse[i-1];
        for (int i = cells-2; i >= 0; --i) if (!coarse[i]) coarse[i] = coarse[i+1];
        int boundary_count = 0, previous = 0;
        for (int knot = 1; knot < cells; ++knot) if (coarse[knot-1] != coarse[knot]) {
            const int centre = 5*knot;
            int begin = centre-4; if (begin < previous+1) begin = previous+1; if (begin < 0) begin = 0;
            int end = centre+4; if (end >= slots) end = slots-1;
            const int local_begin = centre-5 < 0 ? 0 : centre-5;
            const int local_end = centre+4 >= slots ? slots-1 : centre+4;
            /* CONV_LEDGER_COST_BEGIN */
            double increments[10],cost=0.0,energy=0.0;
            for(int slot=local_begin;slot<=local_end;++slot){
                int cell=slot/5,k=slot%5;
                double value=current[(cell*5+k)*lanes+lane],square=value*value;
                int8_t expected=slot<begin?coarse[knot-1]:coarse[knot];
                if(expected*value<0)cost+=square;
                increments[slot-local_begin]=coarse[knot-1]*value<0?square:-square;
                energy+=square;
            }
            double best_cost=cost,second_cost=INFINITY;
            int best_boundary=begin;
            for(int candidate=begin+1;candidate<=end;++candidate){
                cost+=increments[candidate-1-local_begin];
                if(cost<best_cost){second_cost=best_cost;best_cost=cost;best_boundary=candidate;}
                else if(cost<second_cost)second_cost=cost;
            }
            /* At most ten squared samples and eight signed updates.
               An uncertain winner uses the original accumulation and ties. */
            if(!isfinite(energy) || second_cost-best_cost<=64.0*DBL_EPSILON*energy){
                best_cost = INFINITY;
                best_boundary = begin;
                for (int candidate = begin; candidate <= end; ++candidate) {
                    double cost = 0.0;
                    for (int slot = local_begin; slot <= local_end; ++slot) {
                        const int8_t expected = slot < candidate ? coarse[knot-1] : coarse[knot];
                        const int cell = slot / 5, k = slot % 5;
                        const float value = current[(cell*5+k)*lanes+lane];
                        if (expected * value < 0.0f) cost += (double)value * value;
                    }
                    if (cost < best_cost) { best_cost = cost; best_boundary = candidate; }
                }
            }
            /* CONV_LEDGER_COST_END */
            boundaries[boundary_count] = best_boundary;
            new_signs[boundary_count] = coarse[knot];
            ++boundary_count;
            previous = best_boundary;
        }
        int8_t active_sign = coarse[0];
        int next_boundary = 0;
        for (int slot = 0; slot < slots; ++slot) {
            while (next_boundary < boundary_count && slot >= boundaries[next_boundary]) {
                active_sign = new_signs[next_boundary++];
            }
            ledger[slot] = active_sign;
        }
        for (int cell = 0; cell < cells; ++cell) {
            float a[5], c[5]; int8_t signs[5];
            for (int k = 0; k < 5; ++k) {
                a[k] = current[(cell*5+k)*lanes+lane];
                signs[k] = ledger[cell*5+k];
            }
            const float delta = x[(cell+1)*lanes+lane] - x[cell*lanes+lane];
            project_fibre(a, signs, delta, c);
            for (int k = 0; k < 5; ++k) current[(cell*5+k)*lanes+lane] = c[k];
        }
    }
    free(new_signs); free(boundaries); free(ledger); free(coarse);
}

static void ordered_projection(const float *x, int n, int lanes, float *current) {
    projection_context context = {x,n,lanes,current};
    parallel_for(lanes, 24, ordered_projection_worker, &context);
}

static void tail_weights(double u, float tail[5]) {
    const double v = 1.0-u;
    const double b[6] = {
        v*v*v*v*v,
        5.0*u*v*v*v*v,
        10.0*u*u*v*v*v,
        10.0*u*u*u*v*v,
        5.0*u*u*u*u*v,
        u*u*u*u*u
    };
    double sum = 0.0;
    for (int k = 5; k >= 1; --k) { sum += b[k]; tail[k-1] = (float)sum; }
}

typedef struct {
    const float *x;
    const float *current;
    float *out;
    int cells;
    int lanes;
    int m;
} synthesis_context;

static void synthesis_worker(void *opaque, int begin, int end) {
    synthesis_context *job = (synthesis_context *)opaque;
    const float *x=job->x, *current=job->current;
    float *out=job->out;
    const int cells=job->cells, lanes=job->lanes, m=job->m;
    for (int j=begin; j<end; ++j) {
        if (m == 1) {
            memcpy(out, x, (size_t)lanes*sizeof(float));
            continue;
        }
        const double t=(double)j*cells/(double)(m-1);
        const double nearest=nearbyint(t);
        if (fabs(t-nearest)<=4.0e-13 && nearest>=0.0 && nearest<=cells) {
            memcpy(out+(size_t)j*lanes,x+(size_t)((int)nearest)*lanes,
                   (size_t)lanes*sizeof(float));
            continue;
        }
        int cell=(int)floor(t); double u=t-cell;
        if (cell>=cells) { cell=cells-1; u=1.0; }
        float tail[5]; tail_weights(u,tail);
        int lane=0;
#if CONV_NEON
        for (;lane+4<=lanes;lane+=4) {
            float32x4_t value=vld1q_f32(x+cell*lanes+lane);
            for(int k=0;k<5;++k)
                value=vfmaq_n_f32(value,vld1q_f32(current+(cell*5+k)*lanes+lane),tail[k]);
            vst1q_f32(out+j*lanes+lane,value);
        }
#endif
        for(;lane<lanes;++lane){
            float value=x[cell*lanes+lane];
            for(int k=0;k<5;++k)value+=tail[k]*current[(cell*5+k)*lanes+lane];
            out[j*lanes+lane]=value;
        }
    }
}

API int conv_resize_lines_f32(const float *x, int n, int lanes, float *out, int m) {
    if (!x || !out || n < 5 || lanes < 1 || m < 1) return -1;
    const int cells = n-1;
    float *current = (float *)malloc((size_t)cells * 5u * (size_t)lanes * sizeof(float));
    if (!current) return -2;
    raw_currents(x, n, lanes, current);
    ordered_projection(x, n, lanes, current);
    synthesis_context synthesis={x,current,out,cells,lanes,m};
    parallel_for(m,24,synthesis_worker,&synthesis);
    free(current);
    return 0;
}

typedef struct{
    const float*x,*current,*position;float*out;
    int cells,lanes;
}point_context;

static void point_worker(void*opaque,int begin,int end){
    point_context*job=(point_context*)opaque;
    for(int lane=begin;lane<end;++lane){
        double t=job->position[lane];
        if(t<0.0)t=0.0;if(t>job->cells)t=job->cells;
        int cell=(int)floor(t);if(cell>=job->cells)cell=job->cells-1;
        const float u=(float)(t-cell);float tail[5];tail_weights(u,tail);
        float value=job->x[cell*job->lanes+lane];
        for(int k=0;k<5;++k)value+=tail[k]*
            job->current[(cell*5+k)*job->lanes+lane];
        job->out[lane]=value;
    }
}

/* Evaluate one independently positioned sample from every source line. */
API int conv_evaluate_lines_f32(const float*x,int n,int lanes,
                                const float*position,float*out){
    if(!x||!position||!out||n<5||lanes<1)return-1;
    const int cells=n-1;
    float*current=(float*)malloc((size_t)cells*5u*(size_t)lanes*sizeof(float));
    if(!current)return-2;
    raw_currents(x,n,lanes,current);ordered_projection(x,n,lanes,current);
    point_context context={x,current,position,out,cells,lanes};
    parallel_for(lanes,64,point_worker,&context);
    free(current);return 0;
}

typedef struct{
    const float*x,*current,*position;float*out;
    int cells,lanes;
}position_context;

static void position_worker(void*opaque,int begin,int end){
    position_context*job=(position_context*)opaque;
    for(int sample=begin;sample<end;++sample){
        double t=job->position[sample];
        if(t<0.0)t=0.0;if(t>job->cells)t=job->cells;
        int cell=(int)floor(t);if(cell>=job->cells)cell=job->cells-1;
        const float u=(float)(t-cell);float tail[5];tail_weights(u,tail);
        float*destination=job->out+(size_t)sample*job->lanes;
        int lane=0;
#if CONV_NEON
        for(;lane+4<=job->lanes;lane+=4){
            float32x4_t value=vld1q_f32(job->x+cell*job->lanes+lane);
            for(int k=0;k<5;++k)value=vfmaq_n_f32(value,
                vld1q_f32(job->current+(cell*5+k)*job->lanes+lane),tail[k]);
            vst1q_f32(destination+lane,value);
        }
#endif
        for(;lane<job->lanes;++lane){
            float value=job->x[cell*job->lanes+lane];
            for(int k=0;k<5;++k)value+=tail[k]*
                job->current[(cell*5+k)*job->lanes+lane];
            destination[lane]=value;
        }
    }
}

/* Prepared-profile entry points let an already-parallel caller retain the
   current tensor across certification and synthesis.  They deliberately run
   serially: each output is independent, and dispatching the global CONV pool
   from a renderer worker only adds nested scheduling and mutex contention. */
API int conv_prepare_profile_f32(const float *x, int n, int lanes, float *current) {
    if (!x || !current || n < 5 || lanes < 1) return -1;
    raw_currents(x, n, lanes, current);
    ordered_projection(x, n, lanes, current);
    return 0;
}

API int conv_evaluate_prepared_profile_f32(
    const float *x, const float *current, int n, int lanes,
    const float *position, float *out, int m
) {
    if (!x || !current || !position || !out || n < 5 || lanes < 1 || m < 1) return -1;
    position_context context={x,current,position,out,n-1,lanes};
    position_worker(&context,0,m);
    return 0;
}

API int conv_resize_prepared_lines_f32(
    const float *x, const float *current, int n, int lanes, float *out, int m
) {
    if (!x || !current || !out || n < 5 || lanes < 1 || m < 1) return -1;
    synthesis_context synthesis={x,current,out,n-1,lanes,m};
    synthesis_worker(&synthesis,0,m);
    return 0;
}

/* Evaluate all lanes of one profile at an arbitrary ordered site array. */
API int conv_evaluate_profile_f32(const float*x,int n,int lanes,
                                  const float*position,float*out,int m){
    if(!x||!position||!out||n<5||lanes<1||m<1)return-1;
    const int cells=n-1;
    float*current=(float*)malloc((size_t)cells*5u*(size_t)lanes*sizeof(float));
    if(!current)return-2;
    raw_currents(x,n,lanes,current);ordered_projection(x,n,lanes,current);
    position_context context={x,current,position,out,cells,lanes};
    parallel_for(m,24,position_worker,&context);
    free(current);return 0;
}

static void evaluate_precomputed_point_f32(
    const float*x,const float*current,int n,int lanes,double position,float*out){
    const int cells=n-1;double t=position;if(t<0.0)t=0.0;if(t>cells)t=cells;
    int cell=(int)floor(t);if(cell>=cells)cell=cells-1;
    const float u=(float)(t-cell);float tail[5];tail_weights(u,tail);
    for(int lane=0;lane<lanes;++lane){
        float value=x[cell*lanes+lane];
        for(int k=0;k<5;++k)value=fmaf(
            tail[k],current[(cell*5+k)*lanes+lane],value);
        out[lane]=value;
    }
}

typedef struct{
    const float*image,*horizontal,*horizontal_current,*vertical_current,*eta;
    const float*x,*y;float*out;int h,w,channels;
}profile_2d_context;

static void profile_2d_worker(void*opaque,int begin,int end){
    profile_2d_context*job=(profile_2d_context*)opaque;
    const int h=job->h,w=job->w,c=job->channels;
    float*xy_line=(float*)malloc((size_t)h*(size_t)c*sizeof(float));
    float*yx_line=(float*)malloc((size_t)w*(size_t)c*sizeof(float));
    float*xy_current=(float*)malloc((size_t)(h-1)*5u*(size_t)c*sizeof(float));
    float*yx_current=(float*)malloc((size_t)(w-1)*5u*(size_t)c*sizeof(float));
    float*xy_value=(float*)malloc((size_t)c*sizeof(float));
    float*yx_value=(float*)malloc((size_t)c*sizeof(float));
    if(!xy_line||!yx_line||!xy_current||!yx_current||!xy_value||!yx_value){
        free(xy_line);free(yx_line);free(xy_current);free(yx_current);
        free(xy_value);free(yx_value);return;
    }
    for(int sample=begin;sample<end;++sample){
        double x=job->x[sample],y=job->y[sample];
        if(x<0.0)x=0.0;if(x>w-1)x=w-1;if(y<0.0)y=0.0;if(y>h-1)y=h-1;
        evaluate_precomputed_point_f32(
            job->horizontal,job->horizontal_current,w,h*c,x,xy_line);
        raw_currents(xy_line,h,c,xy_current);
        ordered_projection(xy_line,h,c,xy_current);
        evaluate_precomputed_point_f32(xy_line,xy_current,h,c,y,xy_value);

        evaluate_precomputed_point_f32(
            job->image,job->vertical_current,h,w*c,y,yx_line);
        raw_currents(yx_line,w,c,yx_current);
        ordered_projection(yx_line,w,c,yx_current);
        evaluate_precomputed_point_f32(yx_line,yx_current,w,c,x,yx_value);

        int x0=(int)floor(x),y0=(int)floor(y);double fx=x-x0,fy=y-y0;
        if(x0>=w-1){x0=w-2;fx=1.0;}if(y0>=h-1){y0=h-2;fy=1.0;}
        const int x1=x0+1,y1=y0+1;
        const double top=(1.0-fx)*job->eta[(size_t)y0*w+x0]+fx*job->eta[(size_t)y0*w+x1];
        const double bottom=(1.0-fx)*job->eta[(size_t)y1*w+x0]+fx*job->eta[(size_t)y1*w+x1];
        const float beta=(float)((1.0-fy)*top+fy*bottom);
        for(int channel=0;channel<c;++channel){
            const float a=xy_value[channel],b=yx_value[channel];
            job->out[(size_t)sample*c+channel]=a+beta*(b-a);
        }
    }
    free(xy_line);free(yx_line);free(xy_current);free(yx_current);
    free(xy_value);free(yx_value);
}

/* Direct two-dimensional CONV profile evaluation.  Both exact Cartesian
   factor orders are evaluated once through the same source field and blended
   by the parameter-free nodal current coordinate
       beta = Gyy / (Gxx + Gyy).
   Query coordinates are in source-index units. */
API int conv_evaluate_profile_2d_f32(
    const float*image,int h,int w,int channels,
    const float*x,const float*y,float*out,int m){
    if(!image||!x||!y||!out||h<5||w<5||channels<1||m<1)return-1;
    const int horizontal_lanes=h*channels,vertical_lanes=w*channels;
    float*horizontal=(float*)malloc((size_t)w*(size_t)horizontal_lanes*sizeof(float));
    float*horizontal_current=(float*)malloc(
        (size_t)(w-1)*5u*(size_t)horizontal_lanes*sizeof(float));
    float*vertical_current=(float*)malloc(
        (size_t)(h-1)*5u*(size_t)vertical_lanes*sizeof(float));
    float*eta=(float*)malloc((size_t)h*(size_t)w*sizeof(float));
    if(!horizontal||!horizontal_current||!vertical_current||!eta){
        free(horizontal);free(horizontal_current);free(vertical_current);free(eta);return-2;
    }
    for(int sx=0;sx<w;++sx)for(int sy=0;sy<h;++sy)for(int channel=0;channel<channels;++channel)
        horizontal[((size_t)sx*h+sy)*channels+channel]=
            image[((size_t)sy*w+sx)*channels+channel];
    raw_currents(horizontal,w,horizontal_lanes,horizontal_current);
    ordered_projection(horizontal,w,horizontal_lanes,horizontal_current);
    raw_currents(image,h,vertical_lanes,vertical_current);
    ordered_projection(image,h,vertical_lanes,vertical_current);
    for(int sy=0;sy<h;++sy)for(int sx=0;sx<w;++sx){
        double gxx=0.0,gyy=0.0;const float*row=image+(size_t)sy*w*channels;
        for(int channel=0;channel<channels;++channel){
            const double gx=first_jet(row,w,channels,sx,channel);
            const double gy=first_jet(image,h,w*channels,sy,sx*channels+channel);
            gxx+=gx*gx;gyy+=gy*gy;
        }
        const double trace=gxx+gyy;eta[(size_t)sy*w+sx]=(float)(trace>0.0?gyy/trace:0.5);
    }
    profile_2d_context context={
        image,horizontal,horizontal_current,vertical_current,eta,x,y,out,h,w,channels};
    parallel_for(m,32,profile_2d_worker,&context);
    free(horizontal);free(horizontal_current);free(vertical_current);free(eta);return 0;
}

typedef struct{
    const float*x,*current;float*out;
    int cells,lanes,m;
}basin_context;

static void basin_average_worker(void*opaque,int begin,int end){
    basin_context*job=(basin_context*)opaque;
    const double nodes[3]={-0.77459666924148337704,0.0,0.77459666924148337704};
    const double weights[3]={5.0/9.0,8.0/9.0,5.0/9.0};
    const double step=job->m>1?(double)job->cells/(double)(job->m-1):(double)job->cells;
    for(int target=begin;target<end;++target){
        const double left=job->m==1?0.0:(target==0?0.0:(target-0.5)*step);
        const double right=job->m==1?(double)job->cells:
            (target==job->m-1?(double)job->cells:(target+0.5)*step);
        const double width=right-left;
        float*destination=job->out+(size_t)target*job->lanes;
        memset(destination,0,(size_t)job->lanes*sizeof(float));
        int first=(int)floor(left);if(first>=job->cells)first=job->cells-1;
        int last=(int)ceil(right)-1;if(last>=job->cells)last=job->cells-1;
        for(int cell=first;cell<=last;++cell){
            const double segment_left=fmax(left,(double)cell);
            const double segment_right=fmin(right,(double)(cell+1));
            if(segment_right<=segment_left)continue;
            const double middle=0.5*(segment_left+segment_right)-(double)cell;
            const double half=0.5*(segment_right-segment_left);
            /* Compile the exact segment integral once.  Three-point Gauss is
               exact for the quintic profile, but applying every quadrature
               node lane-wise rereads x/current three times.  Accumulating its
               six linear coefficients first reduces the scanline pass to one. */
            float current_factor[5]={0.0f,0.0f,0.0f,0.0f,0.0f};
            const float base_factor=(float)((segment_right-segment_left)/width);
            for(int quadrature=0;quadrature<3;++quadrature){
                const double u=middle+half*nodes[quadrature];
                float tail[5];tail_weights(u,tail);
                const float factor=(float)(half*weights[quadrature]/width);
                for(int k=0;k<5;++k)current_factor[k]+=factor*tail[k];
            }
            int lane=0;
#if CONV_NEON
            for(;lane+4<=job->lanes;lane+=4){
                float32x4_t increment=vmulq_n_f32(
                    vld1q_f32(job->x+cell*job->lanes+lane),base_factor);
                for(int k=0;k<5;++k)increment=vfmaq_n_f32(increment,
                    vld1q_f32(job->current+(cell*5+k)*job->lanes+lane),
                    current_factor[k]);
                vst1q_f32(destination+lane,vaddq_f32(
                    vld1q_f32(destination+lane),increment));
            }
#endif
            for(;lane<job->lanes;++lane){
                float increment=base_factor*job->x[cell*job->lanes+lane];
                for(int k=0;k<5;++k)increment+=current_factor[k]*
                    job->current[(cell*5+k)*job->lanes+lane];
                destination[lane]+=increment;
            }
        }
    }
}

API int conv_basin_average_lines_f32(const float*x,int n,int lanes,float*out,int m){
    if(!x||!out||n<5||lanes<1||m<1||m>n)return-1;
    const int cells=n-1;
    float*current=(float*)malloc((size_t)cells*5u*(size_t)lanes*sizeof(float));
    if(!current)return-2;
    raw_currents(x,n,lanes,current);ordered_projection(x,n,lanes,current);
    basin_context context={x,current,out,cells,lanes,m};
    parallel_for(m,8,basin_average_worker,&context);
    free(current);return 0;
}

/* Double-precision realization of the same ordered-current quintic basin
   operator.  This intentionally keeps a scalar implementation: the endpoint
   data, current ledger, KKT projection, quintic coefficients, and basin
   accumulation all remain binary64 from entry to exit. */
static const double CURRENT_FIR_F64[5][6] = {
    {4.0/240.0, -32.0/240.0, 0.0, 32.0/240.0, -4.0/240.0, 0.0},
    {3.0/240.0, -16.0/240.0, -30.0/240.0, 48.0/240.0, -5.0/240.0, 0.0},
    {-7.0/240.0, 39.0/240.0, -130.0/240.0, 130.0/240.0, -39.0/240.0, 7.0/240.0},
    {0.0, 5.0/240.0, -48.0/240.0, 30.0/240.0, 16.0/240.0, -3.0/240.0},
    {0.0, 4.0/240.0, -32.0/240.0, 0.0, 32.0/240.0, -4.0/240.0}
};

static inline double first_jet_f64(const double*x,int n,int lanes,int i,int lane){
    if(i==0)return(-3.0*x[lane]+4.0*x[lanes+lane]-x[2*lanes+lane])*0.5;
    if(i==1)return(x[2*lanes+lane]-x[lane])*0.5;
    if(i==n-2)return(x[(n-1)*lanes+lane]-x[(n-3)*lanes+lane])*0.5;
    if(i==n-1)return(3.0*x[(n-1)*lanes+lane]-4.0*x[(n-2)*lanes+lane]+x[(n-3)*lanes+lane])*0.5;
    return(x[(i-2)*lanes+lane]-8.0*x[(i-1)*lanes+lane]
           +8.0*x[(i+1)*lanes+lane]-x[(i+2)*lanes+lane])/12.0;
}

static inline double second_jet_f64(const double*x,int n,int lanes,int i,int lane){
    if(i==0)return 2.0*x[lane]-5.0*x[lanes+lane]+4.0*x[2*lanes+lane]-x[3*lanes+lane];
    if(i==1)return x[lane]-2.0*x[lanes+lane]+x[2*lanes+lane];
    if(i==n-2)return x[(n-3)*lanes+lane]-2.0*x[(n-2)*lanes+lane]+x[(n-1)*lanes+lane];
    if(i==n-1)return 2.0*x[(n-1)*lanes+lane]-5.0*x[(n-2)*lanes+lane]
                       +4.0*x[(n-3)*lanes+lane]-x[(n-4)*lanes+lane];
    return(-x[(i+2)*lanes+lane]+16.0*x[(i+1)*lanes+lane]
           -30.0*x[i*lanes+lane]+16.0*x[(i-1)*lanes+lane]
           -x[(i-2)*lanes+lane])/12.0;
}

typedef struct{const double*x;int n,lanes;double*raw;}raw_context_f64;
static void raw_currents_worker_f64(void*opaque,int begin,int end){
    raw_context_f64*job=(raw_context_f64*)opaque;
    const double*x=job->x;const int n=job->n,lanes=job->lanes;double*raw=job->raw;
    for(int cell=begin;cell<end;++cell){
        if(cell>=2&&cell<n-3){
            for(int lane=0;lane<lanes;++lane){
                for(int k=0;k<5;++k){
                    double sum=0.0;
                    for(int tap=0;tap<6;++tap)
                        sum+=CURRENT_FIR_F64[k][tap]*x[(cell-2+tap)*lanes+lane];
                    raw[(cell*5+k)*lanes+lane]=sum;
                }
            }
        }else{
            for(int lane=0;lane<lanes;++lane){
                const double left=x[cell*lanes+lane],right=x[(cell+1)*lanes+lane];
                const double delta=right-left;
                const double m0=first_jet_f64(x,n,lanes,cell,lane);
                const double m1=first_jet_f64(x,n,lanes,cell+1,lane);
                const double q0=second_jet_f64(x,n,lanes,cell,lane);
                const double q1=second_jet_f64(x,n,lanes,cell+1,lane);
                raw[(cell*5+0)*lanes+lane]=m0/5.0;
                raw[(cell*5+1)*lanes+lane]=m0/5.0+q0/20.0;
                raw[(cell*5+2)*lanes+lane]=delta-0.4*(m0+m1)+(q1-q0)/20.0;
                raw[(cell*5+3)*lanes+lane]=m1/5.0-q1/20.0;
                raw[(cell*5+4)*lanes+lane]=m1/5.0;
            }
        }
    }
}

static void raw_currents_f64(const double*x,int n,int lanes,double*raw){
    raw_context_f64 context={x,n,lanes,raw};
    parallel_for(n-1,24,raw_currents_worker_f64,&context);
}

static inline void compare_swap_f64(double*a,double*b){
    if(*a>*b){const double t=*a;*a=*b;*b=t;}
}
static inline void sort_five_f64(double value[5]){
    compare_swap_f64(value+0,value+1);compare_swap_f64(value+3,value+4);
    compare_swap_f64(value+2,value+4);compare_swap_f64(value+2,value+3);
    compare_swap_f64(value+1,value+4);compare_swap_f64(value+0,value+3);
    compare_swap_f64(value+0,value+2);compare_swap_f64(value+1,value+3);
    compare_swap_f64(value+1,value+2);
}

static void simplex_five_f64(const double a[5],double total,double c[5]){
    if(total<=0.0){for(int k=0;k<5;++k)c[k]=0.0;return;}
    double ordered[5];memcpy(ordered,a,sizeof(ordered));sort_five_f64(ordered);
    double prefix=0.0,theta=0.0;
    for(int rank=1;rank<=5;++rank){
        const double value=ordered[5-rank];prefix+=value;
        const double candidate=(prefix-total)/rank;
        if(value>candidate)theta=candidate;
    }
    double mass=0.0;int best=0;
    for(int k=0;k<5;++k){c[k]=fmax(0.0,a[k]-theta);mass+=c[k];if(c[k]>c[best])best=k;}
    c[best]+=total-mass;
}

static void project_fibre_f64(const double a[5],const int8_t sign[5],double delta,double c[5]){
    int all_positive=1,all_negative=1;
    for(int k=0;k<5;++k){all_positive&=sign[k]>0;all_negative&=sign[k]<0;}
    if(all_positive&&delta>=0.0){simplex_five_f64(a,delta,c);return;}
    if(all_negative&&delta<=0.0){
        double reflected[5],projected[5];
        for(int k=0;k<5;++k)reflected[k]=-a[k];
        simplex_five_f64(reflected,-delta,projected);
        for(int k=0;k<5;++k)c[k]=-projected[k];
        return;
    }
    double breaks[5];memcpy(breaks,a,sizeof(breaks));sort_five_f64(breaks);
    int found=0;
    for(int region=0;region<=5&&!found;++region){
        const double low=region==0?-INFINITY:breaks[region-1];
        const double high=region==5?INFINITY:breaks[region];
        double probe;
        if(!isfinite(low))probe=high-fmax(1.0,fabs(high));
        else if(!isfinite(high))probe=low+fmax(1.0,fabs(low));
        else probe=0.5*(low+high);
        int count=0;double sum=0.0;
        for(int k=0;k<5;++k){
            const int active=sign[k]>0?a[k]>probe:a[k]<probe;
            if(active){++count;sum+=a[k];}
        }
        if(!count)continue;
        const double lambda=(sum-delta)/count;
        const double tol=64.0*2.22044604925031308085e-16*fmax(1.0,fmax(fabs(low),fabs(high)));
        if(lambda<low-tol||lambda>high+tol)continue;
        double mass=0.0;
        for(int k=0;k<5;++k){
            const double value=a[k]-lambda;
            c[k]=sign[k]>0?fmax(0.0,value):fmin(0.0,value);mass+=c[k];
        }
        if(fabs(mass-delta)<=256.0*2.22044604925031308085e-16*fmax(1.0,fabs(delta)))found=1;
    }
    if(!found){
        double span=1.0+fabs(delta);
        for(int k=0;k<5;++k)span+=fabs(a[k]);
        double lo=breaks[0]-span,hi=breaks[4]+span;
        for(int it=0;it<80;++it){
            const double lambda=0.5*(lo+hi);double mass=0.0;
            for(int k=0;k<5;++k){
                const double value=a[k]-lambda;
                mass+=sign[k]>0?fmax(0.0,value):fmin(0.0,value);
            }
            if(mass>delta)lo=lambda;else hi=lambda;
        }
        const double lambda=0.5*(lo+hi);
        for(int k=0;k<5;++k){
            const double value=a[k]-lambda;
            c[k]=sign[k]>0?fmax(0.0,value):fmin(0.0,value);
        }
    }
    double mass=0.0,magnitude=-1.0;int best=-1;
    for(int k=0;k<5;++k){
        mass+=c[k];
        if(fabs(c[k])>magnitude){magnitude=fabs(c[k]);best=k;}
    }
    if(best>=0)c[best]+=delta-mass;
}

typedef struct{const double*x;int n,lanes;double*current;}projection_context_f64;
static void ordered_projection_worker_f64(void*opaque,int lane_begin,int lane_end){
    projection_context_f64*job=(projection_context_f64*)opaque;
    const double*x=job->x;const int n=job->n,lanes=job->lanes;
    double*current=job->current;const int cells=n-1,slots=cells*5;
    int8_t*coarse=(int8_t*)malloc((size_t)cells);
    int8_t*ledger=(int8_t*)malloc((size_t)slots);
    int*boundaries=(int*)malloc((size_t)cells*sizeof(int));
    int8_t*new_signs=(int8_t*)malloc((size_t)cells);
    if(!coarse||!ledger||!boundaries||!new_signs)abort();
    for(int lane=lane_begin;lane<lane_end;++lane){
        int any=0;
        for(int cell=0;cell<cells;++cell){
            const double d=x[(cell+1)*lanes+lane]-x[cell*lanes+lane];
            coarse[cell]=d>0.0?1:d<0.0?-1:0;any|=coarse[cell]!=0;
        }
        if(!any){
            for(int slot=0;slot<slots;++slot)current[slot*lanes+lane]=0.0;
            continue;
        }
        for(int i=1;i<cells;++i)if(!coarse[i])coarse[i]=coarse[i-1];
        for(int i=cells-2;i>=0;--i)if(!coarse[i])coarse[i]=coarse[i+1];
        int boundary_count=0,previous=0;
        for(int knot=1;knot<cells;++knot)if(coarse[knot-1]!=coarse[knot]){
            const int centre=5*knot;
            int begin=centre-4;if(begin<previous+1)begin=previous+1;if(begin<0)begin=0;
            int end=centre+4;if(end>=slots)end=slots-1;
            const int local_begin=centre-5<0?0:centre-5;
            const int local_end=centre+4>=slots?slots-1:centre+4;
            double best_cost=INFINITY;int best_boundary=begin;
            for(int candidate=begin;candidate<=end;++candidate){
                double cost=0.0;
                for(int slot=local_begin;slot<=local_end;++slot){
                    const int8_t expected=slot<candidate?coarse[knot-1]:coarse[knot];
                    const int cell=slot/5,k=slot%5;
                    const double value=current[(cell*5+k)*lanes+lane];
                    if(expected*value<0.0)cost+=value*value;
                }
                if(cost<best_cost){best_cost=cost;best_boundary=candidate;}
            }
            boundaries[boundary_count]=best_boundary;new_signs[boundary_count]=coarse[knot];
            ++boundary_count;previous=best_boundary;
        }
        int8_t active_sign=coarse[0];int next_boundary=0;
        for(int slot=0;slot<slots;++slot){
            while(next_boundary<boundary_count&&slot>=boundaries[next_boundary])
                active_sign=new_signs[next_boundary++];
            ledger[slot]=active_sign;
        }
        for(int cell=0;cell<cells;++cell){
            double a[5],c[5];int8_t signs[5];
            for(int k=0;k<5;++k){a[k]=current[(cell*5+k)*lanes+lane];signs[k]=ledger[cell*5+k];}
            const double delta=x[(cell+1)*lanes+lane]-x[cell*lanes+lane];
            project_fibre_f64(a,signs,delta,c);
            for(int k=0;k<5;++k)current[(cell*5+k)*lanes+lane]=c[k];
        }
    }
    free(new_signs);free(boundaries);free(ledger);free(coarse);
}

static void ordered_projection_f64(const double*x,int n,int lanes,double*current){
    projection_context_f64 context={x,n,lanes,current};
    parallel_for(lanes,12,ordered_projection_worker_f64,&context);
}

static void tail_weights_f64(double u,double tail[5]){
    const double v=1.0-u;
    const double b[6]={v*v*v*v*v,5.0*u*v*v*v*v,10.0*u*u*v*v*v,
        10.0*u*u*u*v*v,5.0*u*u*u*u*v,u*u*u*u*u};
    double sum=0.0;for(int k=5;k>=1;--k){sum+=b[k];tail[k-1]=sum;}
}

typedef struct{const double*x,*current;double*out;int cells,lanes,m;}basin_context_f64;
static void basin_average_worker_f64(void*opaque,int begin,int end){
    basin_context_f64*job=(basin_context_f64*)opaque;
    const double nodes[3]={-0.77459666924148337704,0.0,0.77459666924148337704};
    const double weights[3]={5.0/9.0,8.0/9.0,5.0/9.0};
    const double step=job->m>1?(double)job->cells/(double)(job->m-1):(double)job->cells;
    for(int target=begin;target<end;++target){
        const double left=job->m==1?0.0:(target==0?0.0:(target-0.5)*step);
        const double right=job->m==1?(double)job->cells:
            (target==job->m-1?(double)job->cells:(target+0.5)*step);
        const double width=right-left;
        double*destination=job->out+(size_t)target*job->lanes;
        memset(destination,0,(size_t)job->lanes*sizeof(double));
        int first=(int)floor(left);if(first>=job->cells)first=job->cells-1;
        int last=(int)ceil(right)-1;if(last>=job->cells)last=job->cells-1;
        for(int cell=first;cell<=last;++cell){
            const double segment_left=fmax(left,(double)cell);
            const double segment_right=fmin(right,(double)(cell+1));
            if(segment_right<=segment_left)continue;
            const double middle=0.5*(segment_left+segment_right)-(double)cell;
            const double half=0.5*(segment_right-segment_left);
            double current_factor[5]={0.0,0.0,0.0,0.0,0.0};
            const double base_factor=(segment_right-segment_left)/width;
            for(int quadrature=0;quadrature<3;++quadrature){
                const double u=middle+half*nodes[quadrature];double tail[5];
                tail_weights_f64(u,tail);
                const double factor=half*weights[quadrature]/width;
                for(int k=0;k<5;++k)current_factor[k]+=factor*tail[k];
            }
            for(int lane=0;lane<job->lanes;++lane){
                double increment=base_factor*job->x[cell*job->lanes+lane];
                for(int k=0;k<5;++k)increment+=current_factor[k]*job->current[(cell*5+k)*job->lanes+lane];
                destination[lane]+=increment;
            }
        }
    }
}

API int conv_basin_average_lines_f64(const double*x,int n,int lanes,double*out,int m){
    if(!x||!out||n<5||lanes<1||m<1||m>n)return-1;
    const int cells=n-1;
    double*current=(double*)malloc((size_t)cells*5u*(size_t)lanes*sizeof(double));
    if(!current)return-2;
    raw_currents_f64(x,n,lanes,current);ordered_projection_f64(x,n,lanes,current);
    basin_context_f64 context={x,current,out,cells,lanes,m};
    parallel_for(m,4,basin_average_worker_f64,&context);
    free(current);return 0;
}

typedef struct{
    const float*x;const float*current;float*out;
    int n,lanes;float left_tail[5],right_tail[5];
}moment_context;

static void moment_worker(void *opaque,int begin,int end){
    moment_context*job=(moment_context*)opaque;
    const float*x=job->x,*current=job->current;float*out=job->out;
    const int n=job->n,lanes=job->lanes;
    for(int i=begin;i<end;++i){
        if(i==0||i==n-1){
            for(int lane=0;lane<lanes;++lane)
                out[i*lanes+lane]=0.25f*first_jet(x,n,lanes,i,lane);
            continue;
        }
        int lane=0;
#if CONV_NEON
        for(;lane+4<=lanes;lane+=4){
            float32x4_t previous=vld1q_f32(x+(i-1)*lanes+lane);
            float32x4_t next=vld1q_f32(x+i*lanes+lane);
            for(int k=0;k<5;++k){
                previous=vfmaq_n_f32(previous,
                    vld1q_f32(current+((i-1)*5+k)*lanes+lane),job->right_tail[k]);
                next=vfmaq_n_f32(next,
                    vld1q_f32(current+(i*5+k)*lanes+lane),job->left_tail[k]);
            }
            vst1q_f32(out+i*lanes+lane,vmulq_n_f32(vsubq_f32(next,previous),0.5f));
        }
#endif
        for(;lane<lanes;++lane){
            float previous=x[(i-1)*lanes+lane],next=x[i*lanes+lane];
            for(int k=0;k<5;++k){
                previous=fmaf(
                    job->right_tail[k],
                    current[((i-1)*5+k)*lanes+lane], previous
                );
                next=fmaf(
                    job->left_tail[k], current[(i*5+k)*lanes+lane], next
                );
            }
            out[i*lanes+lane]=0.5f*(next-previous);
        }
    }
}

static void moment_lines_core_f32(const float*x,int n,int lanes,float*out,
                                  float*current){
    raw_currents(x,n,lanes,current);ordered_projection(x,n,lanes,current);
    moment_context context={x,current,out,n,lanes,{0},{0}};
    const double nodes[3]={-0.77459666924148337704,0.0,0.77459666924148337704};
    const double weights[3]={5.0/9.0,8.0/9.0,5.0/9.0};
    for(int q=0;q<3;++q){
        float tail[5];
        tail_weights(0.25+0.25*nodes[q],tail);
        for(int k=0;k<5;++k)context.left_tail[k]+=(float)(0.5*weights[q])*tail[k];
        tail_weights(0.75+0.25*nodes[q],tail);
        for(int k=0;k<5;++k)context.right_tail[k]+=(float)(0.5*weights[q])*tail[k];
    }
    parallel_for(n,16,moment_worker,&context);
}

API int conv_moment_lines_f32(const float*x,int n,int lanes,float*out){
    if(!x||!out||n<5||lanes<1)return-1;
    const int cells=n-1;
    float*current=(float*)malloc((size_t)cells*5u*(size_t)lanes*sizeof(float));
    if(!current)return-2;
    moment_lines_core_f32(x,n,lanes,out,current);
    free(current);return 0;
}

static inline float signed_unit(float value){return(float)((value>0.0f)-(value<0.0f));}

typedef struct{
    const float*px,*py,*pxy,*tx,*ty;float*qx,*qy,*qxy;
    int lanes;
}moment_orthant_context;

static void moment_orthant_worker(void*opaque,int begin,int end){
    moment_orthant_context*job=(moment_orthant_context*)opaque;
    for(int item=begin;item<end;++item){
        const float ax=fmaxf(job->tx[item]*job->px[item],0.0f);
        const float ay=fmaxf(job->ty[item]*job->py[item],0.0f);
        job->qx[item]=job->tx[item]*ax;
        job->qy[item]=job->ty[item]*ay;
        job->qxy[item]=signed_unit(job->pxy[item])*
            fminf(fabsf(job->pxy[item]),fminf(ax,ay));
    }
}

static inline float split_face_capacity(float delta,float orientation,
                                         float first,float second){
    if(orientation==0.0f){
        const float sum=first+second;
        const float scale=fabsf(first)+fabsf(second);
        const float gamma8=(8.0f*FLT_EPSILON)/(1.0f-8.0f*FLT_EPSILON);
        return fabsf(sum)<=gamma8*scale?1.0f:0.0f;
    }
    const float consumed=fmaxf(-orientation*first,0.0f)+
                         fmaxf(-orientation*second,0.0f);
    const float base=fabsf(delta);
    return consumed>base&&consumed>0.0f?nextafterf(base/consumed,0.0f):1.0f;
}

typedef struct{
    const float*c,*sx,*sy,*qx,*qy,*qxy;float*ox,*oy,*oxy;
    int height,width,lanes;
}moment_admit_context;

static inline float horizontal_face_capacity(const moment_admit_context*job,
                                              int row,int left,int lane){
    const int lanes=job->lanes,width=job->width;
    const int a=(row*width+left)*lanes+lane;
    const int b=a+lanes;
    const float delta=job->c[b]-job->c[a];
    const float orientation=job->sx[(row*(width-1)+left)*lanes+lane];
    const float upper=split_face_capacity(delta,orientation,
        -job->qx[a]+job->qy[a]+job->qxy[a],
        -job->qx[b]-job->qy[b]+job->qxy[b]);
    const float lower=split_face_capacity(delta,orientation,
        -job->qx[a]-job->qy[a]-job->qxy[a],
        -job->qx[b]+job->qy[b]-job->qxy[b]);
    return fminf(upper,lower);
}

static inline float vertical_face_capacity(const moment_admit_context*job,
                                            int upper_row,int column,int lane){
    const int lanes=job->lanes,width=job->width;
    const int a=(upper_row*width+column)*lanes+lane;
    const int b=a+width*lanes;
    const float delta=job->c[b]-job->c[a];
    const float orientation=job->sy[(upper_row*width+column)*lanes+lane];
    const float left=split_face_capacity(delta,orientation,
        job->qx[a]-job->qy[a]+job->qxy[a],
        -job->qx[b]-job->qy[b]+job->qxy[b]);
    const float right=split_face_capacity(delta,orientation,
        -job->qx[a]-job->qy[a]-job->qxy[a],
        job->qx[b]-job->qy[b]-job->qxy[b]);
    return fminf(left,right);
}

static void moment_admit_worker(void*opaque,int begin,int end){
    moment_admit_context*job=(moment_admit_context*)opaque;
    const int width=job->width,lanes=job->lanes;
    for(int cell=begin;cell<end;++cell){
        const int row=cell/width,column=cell-row*width;
        for(int lane=0;lane<lanes;++lane){
            float alpha=1.0f;
            if(column>0)alpha=fminf(alpha,horizontal_face_capacity(job,row,column-1,lane));
            if(column+1<width)alpha=fminf(alpha,horizontal_face_capacity(job,row,column,lane));
            if(row>0)alpha=fminf(alpha,vertical_face_capacity(job,row-1,column,lane));
            if(row+1<job->height)alpha=fminf(alpha,vertical_face_capacity(job,row,column,lane));
            const int item=cell*lanes+lane;
            job->ox[item]=alpha*job->qx[item];
            job->oy[item]=alpha*job->qy[item];
            job->oxy[item]=alpha*job->qxy[item];
        }
    }
}

static size_t moment_admission_workspace_floats(int height,int width,int lanes){
    const size_t cells=(size_t)height*(size_t)width*(size_t)lanes;
    const size_t horizontal=(size_t)height*(size_t)(width-1)*(size_t)lanes;
    const size_t vertical=(size_t)(height-1)*(size_t)width*(size_t)lanes;
    return horizontal+vertical+5u*cells;
}

static int checked_raster_cells(int height,int width,int lanes,size_t*result){
    const size_t h=(size_t)height,w=(size_t)width,l=(size_t)lanes;
    if(h>SIZE_MAX/w||h*w>SIZE_MAX/l)return 0;
    *result=h*w*l;return 1;
}

static void admit_moments_2d_core_f32(const float*c,const float*px,const float*py,
    const float*pxy,int height,int width,int lanes,float*ox,float*oy,float*oxy,
    float*workspace){
    const size_t cells=(size_t)height*(size_t)width*(size_t)lanes;
    const size_t horizontal=(size_t)height*(size_t)(width-1)*(size_t)lanes;
    const size_t vertical=(size_t)(height-1)*(size_t)width*(size_t)lanes;
    float*sx=workspace;
    float*sy=sx+horizontal;
    float*tx=sy+vertical;float*ty=tx+cells;
    float*qx=ty+cells;float*qy=qx+cells;float*qxy=qy+cells;
    for(int row=0;row<height;++row)for(int lane=0;lane<lanes;++lane){
        for(int column=0;column<width-1;++column){
            const int a=(row*width+column)*lanes+lane;
            sx[(row*(width-1)+column)*lanes+lane]=signed_unit(c[a+lanes]-c[a]);
        }
        for(int column=1;column<width-1;++column){
            float*value=&sx[(row*(width-1)+column)*lanes+lane];
            if(*value==0.0f)*value=sx[(row*(width-1)+column-1)*lanes+lane];
        }
        for(int column=width-3;column>=0;--column){
            float*value=&sx[(row*(width-1)+column)*lanes+lane];
            if(*value==0.0f)*value=sx[(row*(width-1)+column+1)*lanes+lane];
        }
        for(int column=0;column<width;++column){
            const int item=(row*width+column)*lanes+lane;
            const float left=sx[(row*(width-1)+(column?column-1:0))*lanes+lane];
            const float right=sx[(row*(width-1)+(column+1<width?column:width-2))*lanes+lane];
            tx[item]=left==right?left:(px[item]*right>px[item]*left?right:left);
        }
    }
    for(int column=0;column<width;++column)for(int lane=0;lane<lanes;++lane){
        for(int row=0;row<height-1;++row){
            const int a=(row*width+column)*lanes+lane;
            sy[(row*width+column)*lanes+lane]=signed_unit(c[a+width*lanes]-c[a]);
        }
        for(int row=1;row<height-1;++row){
            float*value=&sy[(row*width+column)*lanes+lane];
            if(*value==0.0f)*value=sy[((row-1)*width+column)*lanes+lane];
        }
        for(int row=height-3;row>=0;--row){
            float*value=&sy[(row*width+column)*lanes+lane];
            if(*value==0.0f)*value=sy[((row+1)*width+column)*lanes+lane];
        }
        for(int row=0;row<height;++row){
            const int item=(row*width+column)*lanes+lane;
            const float upper=sy[((row?row-1:0)*width+column)*lanes+lane];
            const float lower=sy[((row+1<height?row:height-2)*width+column)*lanes+lane];
            ty[item]=upper==lower?upper:(py[item]*lower>py[item]*upper?lower:upper);
        }
    }
    moment_orthant_context orthant={px,py,pxy,tx,ty,qx,qy,qxy,lanes};
    parallel_for((int)cells,512,moment_orthant_worker,&orthant);
    moment_admit_context admission={c,sx,sy,qx,qy,qxy,ox,oy,oxy,height,width,lanes};
    parallel_for(height*width,256,moment_admit_worker,&admission);
}

API int conv_admit_moments_2d_f32(const float*c,const float*px,const float*py,
    const float*pxy,int height,int width,int lanes,float*ox,float*oy,float*oxy){
    if(!c||!px||!py||!pxy||!ox||!oy||!oxy||height<2||width<2||lanes<1)return-1;
    size_t cells=0;
    if(!checked_raster_cells(height,width,lanes,&cells)
       ||cells>SIZE_MAX/sizeof(float)/8u)return-1;
    const size_t workspace_floats=moment_admission_workspace_floats(height,width,lanes);
    float*workspace=(float*)malloc(workspace_floats*sizeof(float));
    if(!workspace)return-2;
    admit_moments_2d_core_f32(
        c,px,py,pxy,height,width,lanes,ox,oy,oxy,workspace
    );
    free(workspace);return 0;
}

typedef struct{
    const float*source;float*packed;
    int height,width,lanes;
}moment_pack_horizontal_context;

static void moment_pack_horizontal_worker(void*opaque,int begin,int end){
    moment_pack_horizontal_context*job=(moment_pack_horizontal_context*)opaque;
    const int height=job->height,width=job->width,lanes=job->lanes;
    const int packed_lanes=height*lanes;
    for(int column=begin;column<end;++column){
        for(int row=0;row<height;++row){
            const float*source=job->source+((row*width+column)*lanes);
            float*target=job->packed+column*packed_lanes+row*lanes;
            memcpy(target,source,(size_t)lanes*sizeof(float));
        }
    }
}

typedef struct{
    const float*packed;float*target;
    int height,width,lanes;
}moment_unpack_horizontal_context;

static void moment_unpack_horizontal_worker(void*opaque,int begin,int end){
    moment_unpack_horizontal_context*job=(moment_unpack_horizontal_context*)opaque;
    const int height=job->height,width=job->width,lanes=job->lanes;
    const int packed_lanes=height*lanes;
    for(int column=begin;column<end;++column){
        for(int row=0;row<height;++row){
            const float*source=job->packed+column*packed_lanes+row*lanes;
            float*target=job->target+((row*width+column)*lanes);
            memcpy(target,source,(size_t)lanes*sizeof(float));
        }
    }
}

typedef struct{
    const float*coarse,*horizontal,*vertical,*mixed;float*fine;
    int width,lanes;
}moment_atlas_synthesis_context;

static void moment_atlas_synthesis_worker(void*opaque,int begin,int end){
    moment_atlas_synthesis_context*job=(moment_atlas_synthesis_context*)opaque;
    const int width=job->width,lanes=job->lanes,fine_width=2*width;
    for(int cell=begin;cell<end;++cell){
        const int row=cell/width,column=cell-row*width;
        const int source=cell*lanes;
        const int upper_left=((2*row)*fine_width+2*column)*lanes;
        const int upper_right=upper_left+lanes;
        const int lower_left=upper_left+fine_width*lanes;
        const int lower_right=lower_left+lanes;
        int lane=0;
#if CONV_NEON
        for(;lane+4<=lanes;lane+=4){
            const float32x4_t mean=vld1q_f32(job->coarse+source+lane);
            const float32x4_t qx=vld1q_f32(job->horizontal+source+lane);
            const float32x4_t qy=vld1q_f32(job->vertical+source+lane);
            const float32x4_t qxy=vld1q_f32(job->mixed+source+lane);
            const float32x4_t minus_x=vsubq_f32(mean,qx);
            const float32x4_t plus_x=vaddq_f32(mean,qx);
            vst1q_f32(job->fine+upper_left+lane,
                vaddq_f32(vsubq_f32(minus_x,qy),qxy));
            vst1q_f32(job->fine+upper_right+lane,
                vsubq_f32(vsubq_f32(plus_x,qy),qxy));
            vst1q_f32(job->fine+lower_left+lane,
                vsubq_f32(vaddq_f32(minus_x,qy),qxy));
            vst1q_f32(job->fine+lower_right+lane,
                vaddq_f32(vaddq_f32(plus_x,qy),qxy));
        }
#endif
        for(;lane<lanes;++lane){
            const float mean=job->coarse[source+lane];
            const float qx=job->horizontal[source+lane];
            const float qy=job->vertical[source+lane];
            const float qxy=job->mixed[source+lane];
            job->fine[upper_left+lane]=mean-qx-qy+qxy;
            job->fine[upper_right+lane]=mean+qx-qy-qxy;
            job->fine[lower_left+lane]=mean-qx+qy-qxy;
            job->fine[lower_right+lane]=mean+qx+qy+qxy;
        }
    }
}

/*
 * Form the complete conservative four-child moment atlas in one native call.
 * Horizontal proposals are evaluated as a packed line bank.  Vertical and
 * mixed proposals retain the raster's already-native line-bank layout, after
 * which complete face-current admission is applied in place and the four
 * children are written directly to their unique fine-grid cells.
 */
API int conv_four_child_moment_atlas_f32(const float*coarse,int height,int width,
                                         int lanes,float*fine){
    if(!coarse||!fine||height<5||width<5||lanes<1)return-1;
    if(lanes>INT_MAX/2||height>INT_MAX/lanes||width>INT_MAX/(2*lanes))return-1;
    size_t cells=0;
    if(!checked_raster_cells(height,width,lanes,&cells)
       ||cells>SIZE_MAX/sizeof(float)/16u)return-1;
    const size_t horizontal_current=5u*(size_t)(width-1)*(size_t)height*(size_t)lanes;
    const size_t vertical_current=5u*(size_t)(height-1)*(size_t)width*(size_t)lanes;
    const size_t current_floats=horizontal_current>vertical_current?
        horizontal_current:vertical_current;
    const size_t moment_scratch=2u*cells+current_floats;
    const size_t admission_scratch=moment_admission_workspace_floats(height,width,lanes);
    const size_t scratch_floats=moment_scratch>admission_scratch?
        moment_scratch:admission_scratch;
    float*storage=(float*)malloc((3u*cells+scratch_floats)*sizeof(float));
    if(!storage)return-2;
    float*horizontal=storage;
    float*vertical=horizontal+cells;
    float*mixed=vertical+cells;
    float*scratch=mixed+cells;
    float*packed=scratch;
    float*packed_moments=packed+cells;
    float*current=packed_moments+cells;

    moment_pack_horizontal_context horizontal_pack={coarse,packed,height,width,lanes};
    parallel_for(width,8,moment_pack_horizontal_worker,&horizontal_pack);
    moment_lines_core_f32(packed,width,height*lanes,packed_moments,current);
    moment_unpack_horizontal_context horizontal_unpack={
        packed_moments,horizontal,height,width,lanes
    };
    parallel_for(width,8,moment_unpack_horizontal_worker,&horizontal_unpack);

    /* Rows already form the native vertical line-bank layout.  Keeping the
       two inputs in place avoids a 4*cells pack/unpack traffic round trip. */
    moment_lines_core_f32(coarse,height,width*lanes,vertical,current);
    moment_lines_core_f32(horizontal,height,width*lanes,mixed,current);

    admit_moments_2d_core_f32(
        coarse,horizontal,vertical,mixed,height,width,lanes,
        horizontal,vertical,mixed,scratch
    );
    moment_atlas_synthesis_context synthesis={
        coarse,horizontal,vertical,mixed,fine,width,lanes
    };
    parallel_for(height*width,256,moment_atlas_synthesis_worker,&synthesis);
    free(storage);return 0;
}

static inline double sinc_pi(double x) {
    if (fabs(x) < 1.0e-12) return 1.0;
    const double p = 3.14159265358979323846264338327950288 * x;
    return sin(p) / p;
}

typedef struct {
    const float *x;
    float *out;
    int n,lanes,m,radius;
    double cutoff,support;
} fir_context;

static void fir_worker(void *opaque,int first,int last){
    fir_context *job=(fir_context *)opaque;
    const float *x=job->x; float *out=job->out;
    const int n=job->n,lanes=job->lanes,m=job->m;
    const double cutoff=job->cutoff,support=job->support;
    for (int j=first;j<last;++j) {
        const double t = m == 1 ? 0.0 : (double)j*(n-1)/(double)(m-1);
        const int begin = (int)ceil(t-support), end = (int)floor(t+support);
        double normalizer = 0.0;
        for (int tap = begin; tap <= end; ++tap) {
            const double d = t-tap;
            normalizer += cutoff*sinc_pi(cutoff*d)*sinc_pi(d/support);
        }
        if (fabs(normalizer) < 1.0e-18) normalizer = 1.0;
        int lane = 0;
#if CONV_NEON
        for (; lane + 4 <= lanes; lane += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int tap = begin; tap <= end; ++tap) {
                int index = tap < 0 ? 0 : tap >= n ? n-1 : tap;
                const double d = t-tap;
                const float w = (float)(cutoff*sinc_pi(cutoff*d)*sinc_pi(d/support)/normalizer);
                acc = vfmaq_n_f32(acc, vld1q_f32(x+index*lanes+lane), w);
            }
            vst1q_f32(out+j*lanes+lane, acc);
        }
#endif
        for (; lane < lanes; ++lane) {
            double acc = 0.0;
            for (int tap = begin; tap <= end; ++tap) {
                int index = tap < 0 ? 0 : tap >= n ? n-1 : tap;
                const double d = t-tap;
                const double w = cutoff*sinc_pi(cutoff*d)*sinc_pi(d/support)/normalizer;
                acc += w*x[index*lanes+lane];
            }
            out[j*lanes+lane] = (float)acc;
        }
    }
}

API int fir_resize_lines_f32(const float *x, int n, int lanes, float *out, int m, int radius) {
    if (!x || !out || n < 1 || lanes < 1 || m < 1 || radius < 1) return -1;
    const double rate=m>1&&n>1?(double)(m-1)/(double)(n-1):1.0;
    const double cutoff=rate<1.0?rate:1.0;
    fir_context context={x,out,n,lanes,m,radius,cutoff,radius/cutoff};
    parallel_for(m,32,fir_worker,&context);
    return 0;
}

typedef struct{const float*x;float*out;int n,lanes,m;}linear_context;

static void linear_worker(void *opaque,int first,int last){
    linear_context *job=(linear_context *)opaque;
    const float*x=job->x;float*out=job->out;
    const int n=job->n,lanes=job->lanes,m=job->m;
    for(int j=first;j<last;++j){
        const double t = m == 1 ? 0.0 : (double)j*(n-1)/(double)(m-1);
        int left = (int)floor(t); if (left >= n-1) left = n-1;
        int right = left < n-1 ? left+1 : left;
        const float u = (float)(t-left);
        int lane=0;
#if CONV_NEON
        const float32x4_t vu=vdupq_n_f32(u),vv=vdupq_n_f32(1.0f-u);
        for(;lane+4<=lanes;lane+=4){
            const float32x4_t a=vld1q_f32(x+left*lanes+lane);
            const float32x4_t b=vld1q_f32(x+right*lanes+lane);
            vst1q_f32(out+j*lanes+lane,vfmaq_f32(vmulq_f32(vv,a),vu,b));
        }
#endif
        for(;lane<lanes;++lane)
            out[j*lanes+lane] = (1.0f-u)*x[left*lanes+lane] + u*x[right*lanes+lane];
    }
}

API int linear_resize_lines_f32(const float *x, int n, int lanes, float *out, int m) {
    if (!x || !out || n < 1 || lanes < 1 || m < 1) return -1;
    linear_context context={x,out,n,lanes,m};
    parallel_for(m,64,linear_worker,&context);
    return 0;
}

static inline float evaluate_admitted_lane(const float *x,const float *current,
                                            int cells,int lanes,int lane,double t){
    if(t<0.0)t=0.0;if(t>cells)t=cells;
    const double nearest=nearbyint(t);
    if(fabs(t-nearest)<=4.0e-13&&nearest>=0.0&&nearest<=cells)
        return x[(size_t)((int)nearest)*lanes+lane];
    int cell=(int)floor(t);if(cell>=cells)cell=cells-1;
    const double u=t-cell;float tail[5];tail_weights(u,tail);
    float value=x[(size_t)cell*lanes+lane];
    for(int k=0;k<5;++k)value+=tail[k]*
        current[((size_t)cell*5u+(size_t)k)*lanes+lane];
    return value;
}

typedef struct{
    const float*image,*baseline,*row_x,*row_current,*column_current;
    const float*mass,*tx,*ty;
    float*out;
    int h,w,channels,oh,ow;
}oriented_chord_context;

static inline float oriented_boundary_value(
    const oriented_chord_context*job,double x,double y,int horizontal,int channel){
    if(horizontal){
        int row=(int)nearbyint(y);if(row<0)row=0;if(row>=job->h)row=job->h-1;
        const int lane=row*job->channels+channel;
        return evaluate_admitted_lane(job->row_x,job->row_current,
                                      job->w-1,job->h*job->channels,lane,x);
    }
    int column=(int)nearbyint(x);if(column<0)column=0;if(column>=job->w)column=job->w-1;
    const int lane=column*job->channels+channel;
    return evaluate_admitted_lane(job->image,job->column_current,
                                  job->h-1,job->w*job->channels,lane,y);
}

static void oriented_chord_worker(void*opaque,int row_begin,int row_end){
    oriented_chord_context*job=(oriented_chord_context*)opaque;
    const double epsilon=128.0*2.22044604925031308085e-16;
    for(int oy=row_begin;oy<row_end;++oy){
        const double sy=job->oh==1?0.0:(double)oy*(job->h-1)/(double)(job->oh-1);
        int iy=(int)floor(sy);if(iy>=job->h-1)iy=job->h-2;
        for(int ox=0;ox<job->ow;++ox){
            const double sx=job->ow==1?0.0:(double)ox*(job->w-1)/(double)(job->ow-1);
            int ix=(int)floor(sx);if(ix>=job->w-1)ix=job->w-2;
            const int cell=iy*(job->w-1)+ix;
            float mass=job->mass[cell];
            const double tangent_x=job->tx[cell],tangent_y=job->ty[cell];
            const int nested=fabs(sx-nearbyint(sx))<=epsilon&&
                             fabs(sy-nearbyint(sy))<=epsilon;
            double endpoint_x[2],endpoint_y[2],length[2];int horizontal[2];
            for(int side=0;side<2;++side){
                const double direction=side?1.0:-1.0;
                const double dx=direction*tangent_x,dy=direction*tangent_y;
                double xd=INFINITY,yd=INFINITY;
                if(dx>epsilon)xd=(ix+1.0-sx)/dx;
                else if(dx<-epsilon)xd=(ix-sx)/dx;
                if(dy>epsilon)yd=(iy+1.0-sy)/dy;
                else if(dy<-epsilon)yd=(iy-sy)/dy;
                horizontal[side]=yd<=xd;
                length[side]=fmin(xd,yd);
                endpoint_x[side]=sx+length[side]*dx;
                endpoint_y[side]=sy+length[side]*dy;
            }
            const double denominator=length[0]+length[1];
            if(nested||!(denominator>epsilon)||!isfinite(denominator))mass=0.0f;
            const size_t output=((size_t)oy*job->ow+ox)*job->channels;
            if(mass<=0.0f){
                memcpy(job->out+output,job->baseline+output,
                       (size_t)job->channels*sizeof(float));
                continue;
            }
            for(int channel=0;channel<job->channels;++channel){
                const float backward=oriented_boundary_value(
                    job,endpoint_x[0],endpoint_y[0],horizontal[0],channel);
                const float forward=oriented_boundary_value(
                    job,endpoint_x[1],endpoint_y[1],horizontal[1],channel);
                const double chord=(length[1]*backward+length[0]*forward)/denominator;
                job->out[output+channel]=(1.0f-mass)*job->baseline[output+channel]
                    +mass*(float)chord;
            }
        }
    }
}

/* Blend a symmetric Cartesian CONV baseline with the bounded tangent chord.
   Geometry is the multichannel first-current tensor summed over the exact
   six-by-six two-jet read of each source cell. */
API int conv_oriented_chord_blend_f32(const float*image,int h,int w,int channels,
    const float*baseline,float*out,int oh,int ow){
    if(!image||!baseline||!out||h<5||w<5||channels<1||oh<1||ow<1)return-1;
    const size_t nodes=(size_t)h*(size_t)w;
    const size_t integral_nodes=(size_t)(h+1)*(size_t)(w+1);
    const size_t cells=(size_t)(h-1)*(size_t)(w-1);
    double*gxx=(double*)calloc(integral_nodes,sizeof(double));
    double*gxy=(double*)calloc(integral_nodes,sizeof(double));
    double*gyy=(double*)calloc(integral_nodes,sizeof(double));
    float*mass=(float*)malloc(cells*sizeof(float));
    float*tx=(float*)malloc(cells*sizeof(float));
    float*ty=(float*)malloc(cells*sizeof(float));
    const int row_lanes=h*channels,column_lanes=w*channels;
    float*row_x=(float*)malloc((size_t)w*(size_t)row_lanes*sizeof(float));
    float*row_current=(float*)malloc((size_t)(w-1)*5u*(size_t)row_lanes*sizeof(float));
    float*column_current=(float*)malloc((size_t)(h-1)*5u*(size_t)column_lanes*sizeof(float));
    if(!gxx||!gxy||!gyy||!mass||!tx||!ty||!row_x||!row_current||!column_current){
        free(gxx);free(gxy);free(gyy);free(mass);free(tx);free(ty);
        free(row_x);free(row_current);free(column_current);return-2;
    }
    (void)nodes;
    for(int y=0;y<h;++y){
        double row_xx=0.0,row_xy=0.0,row_yy=0.0;
        for(int x=0;x<w;++x){
            double xx=0.0,xy=0.0,yy=0.0;
            const float*row=image+(size_t)y*w*channels;
            for(int channel=0;channel<channels;++channel){
                const float gx=first_jet(row,w,channels,x,channel);
                const float gy=first_jet(image,h,w*channels,y,x*channels+channel);
                xx+=(double)gx*gx;xy+=(double)gx*gy;yy+=(double)gy*gy;
                row_x[((size_t)x*row_lanes)+(size_t)y*channels+channel]=
                    image[((size_t)y*w+x)*channels+channel];
            }
            row_xx+=xx;row_xy+=xy;row_yy+=yy;
            const size_t here=(size_t)(y+1)*(w+1)+(x+1);
            const size_t above=(size_t)y*(w+1)+(x+1);
            gxx[here]=gxx[above]+row_xx;
            gxy[here]=gxy[above]+row_xy;
            gyy[here]=gyy[above]+row_yy;
        }
    }
    for(int y=0;y<h-1;++y)for(int x=0;x<w-1;++x){
        const int top=y>2?y-2:0,left=x>2?x-2:0;
        const int bottom=y+3<h?y+3:h-1,right=x+3<w?x+3:w-1;
        const size_t br=(size_t)(bottom+1)*(w+1)+(right+1);
        const size_t tr=(size_t)top*(w+1)+(right+1);
        const size_t bl=(size_t)(bottom+1)*(w+1)+left;
        const size_t tl=(size_t)top*(w+1)+left;
        const double xx=gxx[br]-gxx[tr]-gxx[bl]+gxx[tl];
        const double xy=gxy[br]-gxy[tr]-gxy[bl]+gxy[tl];
        const double yy=gyy[br]-gyy[tr]-gyy[bl]+gyy[tl];
        const double trace=xx+yy;
        const double gap=sqrt(fmax((xx-yy)*(xx-yy)+4.0*xy*xy,0.0));
        const double angle=0.5*atan2(2.0*xy,xx-yy);
        const int cell=y*(w-1)+x;
        mass[cell]=(float)(trace>0.0?fmin(1.0,fmax(0.0,gap/trace)):0.0);
        tx[cell]=(float)-sin(angle);ty[cell]=(float)cos(angle);
    }
    raw_currents(row_x,w,row_lanes,row_current);
    ordered_projection(row_x,w,row_lanes,row_current);
    raw_currents(image,h,column_lanes,column_current);
    ordered_projection(image,h,column_lanes,column_current);
    oriented_chord_context context={image,baseline,row_x,row_current,column_current,
        mass,tx,ty,out,h,w,channels,oh,ow};
    parallel_for(oh,8,oriented_chord_worker,&context);
    free(gxx);free(gxy);free(gyy);free(mass);free(tx);free(ty);
    free(row_x);free(row_current);free(column_current);return 0;
}

typedef struct {
    const float *eta;
    const float *forward;
    const float *reverse;
    float *out;
    int h,w,channels,oh,ow;
} q1_blend_context;

static void q1_blend_worker(void *opaque, int begin, int end) {
    q1_blend_context *job=(q1_blend_context *)opaque;
    for(int oy=begin;oy<end;++oy){
        const double sy=job->oh==1?0.0:(double)oy*(job->h-1)/(job->oh-1);
        int y0=(int)floor(sy);double fy=sy-y0;
        if(y0>=job->h-1){y0=job->h-2;fy=1.0;}
        const int y1=y0+1;
        for(int ox=0;ox<job->ow;++ox){
            const double sx=job->ow==1?0.0:(double)ox*(job->w-1)/(job->ow-1);
            int x0=(int)floor(sx);double fx=sx-x0;
            if(x0>=job->w-1){x0=job->w-2;fx=1.0;}
            const int x1=x0+1;
            const double top=(1.0-fx)*job->eta[(size_t)y0*job->w+x0]
                             +fx*job->eta[(size_t)y0*job->w+x1];
            const double bottom=(1.0-fx)*job->eta[(size_t)y1*job->w+x0]
                                +fx*job->eta[(size_t)y1*job->w+x1];
            const float beta=(float)((1.0-fy)*top+fy*bottom);
            const size_t base=((size_t)oy*job->ow+ox)*job->channels;
            int channel=0;
#if CONV_NEON
            for(;channel+4<=job->channels;channel+=4){
                const float32x4_t a=vld1q_f32(job->forward+base+channel);
                const float32x4_t b=vld1q_f32(job->reverse+base+channel);
                vst1q_f32(job->out+base+channel,
                          vfmaq_n_f32(a,vsubq_f32(b,a),beta));
            }
#endif
            for(;channel<job->channels;++channel){
                const float a=job->forward[base+channel];
                job->out[base+channel]=a+beta*(job->reverse[base+channel]-a);
            }
        }
    }
}

/* Evaluate the cardinal Q1 order coordinate and the final two-order convex
   blend in one output pass.  No target beta raster or separable beta resize
   is materialized. */
API int conv_q1_order_blend_f32(const float *eta,int h,int w,
    const float *forward,const float *reverse,int channels,
    float *out,int oh,int ow){
    if(!eta||!forward||!reverse||!out||h<2||w<2||channels<1||oh<1||ow<1)
        return-1;
    q1_blend_context context={eta,forward,reverse,out,h,w,channels,oh,ow};
    parallel_for(oh,8,q1_blend_worker,&context);
    return 0;
}

static inline const float *pixel_at(const float *image, int h, int w, int c, int x, int y) {
    if (x < 0) x = 0; else if (x >= w) x = w-1;
    if (y < 0) y = 0; else if (y >= h) y = h-1;
    return image + ((size_t)y*w + x)*c;
}

static inline float easu_luma(const float *p, int c) {
    if (c == 1) return 2.0f*p[0];
    return 0.5f*p[0] + p[1] + 0.5f*p[2];
}

static float easu_direction(float dir[2], float length, float weight,
                            float la, float lb, float lc, float ld, float le) {
    const float dc = ld-lc, cb = lc-lb;
    const float denx = fmaxf(fabsf(dc), fabsf(cb));
    const float dx = ld-lb;
    dir[0] += dx*weight;
    const float lx = denx == 0.0f ? 0.0f : fminf(1.0f, fabsf(dx)/denx);
    const float ec = le-lc, ca = lc-la;
    const float deny = fmaxf(fabsf(ec), fabsf(ca));
    const float dy = le-la;
    dir[1] += dy*weight;
    const float ly = deny == 0.0f ? 0.0f : fminf(1.0f, fabsf(dy)/deny);
    return length + weight*(lx*lx+ly*ly);
}

typedef struct {
    const float *image;
    float *out;
    int h,w,channels,oh,ow;
} easu_context;

static void easu_worker(void *opaque,int row_begin,int row_end){
    easu_context *job=(easu_context *)opaque;
    const float *image=job->image;float*out=job->out;
    const int h=job->h,w=job->w,channels=job->channels,oh=job->oh,ow=job->ow;
    static const int dx[12] = {0,1,-1,0,1,2,-1,0,1,2,0,1};
    static const int dy[12] = {-1,-1,0,0,0,0,1,1,1,1,2,2};
    static const int group[4][5] = {
        {0,2,3,4,7}, {1,3,4,5,8}, {3,6,7,8,10}, {4,7,8,9,11}
    };
    for (int oy=row_begin;oy<row_end;++oy) {
        const float sy = oh == 1 ? 0.0f : (float)oy*(h-1)/(float)(oh-1);
        const int by = (int)floorf(sy); const float fy = sy-by;
        for (int ox = 0; ox < ow; ++ox) {
            const float sx = ow == 1 ? 0.0f : (float)ox*(w-1)/(float)(ow-1);
            const int bx = (int)floorf(sx); const float fx = sx-bx;
            const float *tap[12]; float luma[12];
            for (int k = 0; k < 12; ++k) {
                tap[k] = pixel_at(image,h,w,channels,bx+dx[k],by+dy[k]);
                luma[k] = easu_luma(tap[k],channels);
            }
            const float blend[4] = {
                (1.0f-fx)*(1.0f-fy), fx*(1.0f-fy),
                (1.0f-fx)*fy, fx*fy
            };
            float dir[2] = {0.0f,0.0f}, edge = 0.0f;
            for (int q = 0; q < 4; ++q) {
                const int *g = group[q];
                edge = easu_direction(dir,edge,blend[q],luma[g[0]],luma[g[1]],
                                      luma[g[2]],luma[g[3]],luma[g[4]]);
            }
            const float norm2 = dir[0]*dir[0]+dir[1]*dir[1];
            if (norm2 < 1.0f/32768.0f) { dir[0]=1.0f; dir[1]=0.0f; }
            else { const float inv=1.0f/sqrtf(norm2); dir[0]*=inv; dir[1]*=inv; }
            edge = 0.25f*edge*edge;
            const float stretch = 1.0f/fmaxf(fabsf(dir[0]),fabsf(dir[1]));
            const float ls0 = 1.0f+(stretch-1.0f)*edge, ls1 = 1.0f-0.5f*edge;
            const float lobe = 0.5f+((0.25f-0.04f)-0.5f)*edge;
            const float clip2 = 1.0f/lobe;
            float weights[12], total = 0.0f;
            for (int k = 0; k < 12; ++k) {
                const float px=dx[k]-fx, py=dy[k]-fy;
                const float vx=(px*dir[0]+py*dir[1])*ls0;
                const float vy=(-px*dir[1]+py*dir[0])*ls1;
                const float d2=fminf(vx*vx+vy*vy,clip2);
                float wb=0.4f*d2-1.0f, wa=lobe*d2-1.0f;
                wb=1.5625f*wb*wb-0.5625f;
                weights[k]=wb*wa*wa; total+=weights[k];
            }
            float *dst=out+((size_t)oy*ow+ox)*channels;
            for (int ch=0; ch<channels; ++ch) {
                float value=0.0f;
                for (int k=0;k<12;++k) value+=tap[k][ch]*weights[k];
                value/=total;
                float low=tap[3][ch], high=tap[3][ch];
                const int nearest[3]={4,7,8};
                for(int k=0;k<3;++k){float v=tap[nearest[k]][ch];low=fminf(low,v);high=fmaxf(high,v);}
                dst[ch]=fminf(high,fmaxf(low,value));
            }
        }
    }
}

API int easu_resize_f32(const float *image, int h, int w, int channels,
                        float *out, int oh, int ow) {
    if (!image || !out || h < 1 || w < 1 || channels < 1 || oh < 1 || ow < 1) return -1;
    easu_context context={image,out,h,w,channels,oh,ow};
    parallel_for(oh,16,easu_worker,&context);
    return 0;
}

/* CONV_TERMINAL_FUSION_BEGIN */
/* Exact terminal dataflow fusion. Both original admitted profiles are input. */
typedef struct { int cell,anchor; float t[5]; } fused_phase;
typedef struct {
 const float *x,*y,*cy,*cx,*eta;float *out;const double *etax;
 int h,w,oh,ow,c;const fused_phase *px,*py;
} fused_terminal_context;
static void fused_terminal_worker(void *opaque,int begin,int end){
 fused_terminal_context *j=(fused_terminal_context *)opaque;
 const int lx=j->ow*j->c,ly=j->oh*j->c;
 for(int oy=begin;oy<end;++oy){
  const fused_phase *p=j->py+oy;
  double sy=j->oh==1?0.0:(double)oy*(j->h-1)/(j->oh-1);
  int y0=(int)floor(sy);double fy=sy-y0;
  if(y0>=j->h-1){y0=j->h-2;fy=1.0;}
  for(int ox=0;ox<j->ow;++ox){
   const fused_phase *q=j->px+ox;
   const double top=j->etax[y0*j->ow+ox];
   const double bottom=j->etax[(y0+1)*j->ow+ox];
   float beta=(float)((1.0-fy)*top+fy*bottom);
   for(int ch=0;ch<j->c;++ch){
    int a_lane=ox*j->c+ch,b_lane=oy*j->c+ch;
    float a=j->x[(size_t)(p->anchor>=0?p->anchor:p->cell)*lx+a_lane];
    float b=j->y[(size_t)(q->anchor>=0?q->anchor:q->cell)*ly+b_lane];
    if(p->anchor<0)for(int k=0;k<5;++k){
     float z=j->cy[(size_t)(p->cell*5+k)*lx+a_lane];
#if CONV_NEON
     if(a_lane<lx-lx%4)a=fmaf(p->t[k],z,a);else
#endif
     a+=p->t[k]*z;
    }
    if(q->anchor<0)for(int k=0;k<5;++k){
     float z=j->cx[(size_t)(q->cell*5+k)*ly+b_lane];
#if CONV_NEON
     if(b_lane<ly-ly%4)b=fmaf(q->t[k],z,b);else
#endif
     b+=q->t[k]*z;
    }
#if CONV_NEON
    if(ch<j->c-j->c%4)j->out[(size_t)oy*lx+a_lane]=fmaf(beta,b-a,a);else
#endif
    j->out[(size_t)oy*lx+a_lane]=a+beta*(b-a);
   }
  }
 }
}
static void fused_make_phases(fused_phase *p,int n,int m){
 for(int i=0;i<m;++i){
  double t=m==1?0.0:(double)i*(n-1)/(m-1);
  int anchor=(int)llround(t);p[i].anchor=-1;
  if(anchor>=0&&anchor<n&&fabs(t-anchor)<=4.0e-13){p[i].anchor=anchor;p[i].cell=anchor;continue;}
  int cell=(int)floor(t);if(cell>=n-1)cell=n-2;
  p[i].cell=cell;tail_weights(t-cell,p[i].t);
 }
}
API int conv_fused_terminal(const float *x,const float *y,const float *cy,const float *cx,
 const float *eta,int h,int w,int oh,int ow,int c,float *out){
 if(!x||!y||!cy||!cx||!eta||!out||h<5||w<5||oh<1||ow<1||c<1)return -2;
 fused_phase *px=malloc((size_t)ow*sizeof(*px)),*py=malloc((size_t)oh*sizeof(*py));
 if(!px||!py){free(px);free(py);return -1;}
 fused_make_phases(px,w,ow);fused_make_phases(py,h,oh);
 double *etax=malloc((size_t)h*ow*sizeof(*etax));
 if(!etax){free(px);free(py);return -1;}
 for(int ox=0;ox<ow;++ox){
  double sx=ow==1?0.0:(double)ox*(w-1)/(ow-1);
  int x0=(int)floor(sx);double fx=sx-x0;
  if(x0>=w-1){x0=w-2;fx=1.0;}
  for(int iy=0;iy<h;++iy)
   etax[(size_t)iy*ow+ox]=(1.0-fx)*eta[(size_t)iy*w+x0]+fx*eta[(size_t)iy*w+x0+1];
 }
 fused_terminal_context j={x,y,cy,cx,eta,out,etax,h,w,oh,ow,c,px,py};
 parallel_for(oh,8,fused_terminal_worker,&j);free(etax);free(px);free(py);return 0;
}
/* CONV_TERMINAL_FUSION_END */

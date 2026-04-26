#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifndef ML_ARENA_DEFAULT_CAPACITY
#  define ML_ARENA_DEFAULT_CAPACITY  (64 * 1024 * 1024) 
#endif

#ifndef ML_DEBUG_MAX_ALLOCS
#  define ML_DEBUG_MAX_ALLOCS  65536
#endif

void* ml_malloc(size_t size);

void* ml_calloc(size_t n, size_t size);

void* ml_realloc(void *ptr, size_t new_size);

void  ml_free(void *ptr);

char* ml_strdup(const char *s);

typedef struct {
    unsigned char *buf; 
    size_t         cap;    
    size_t         used;   
    size_t         peak;    
} MlArena;

MlArena* arena_create(size_t capacity);
void     arena_free(MlArena *a);
void*    arena_alloc(MlArena *a, size_t size);
void*    arena_calloc(MlArena *a, size_t n, size_t size);
void     arena_reset(MlArena *a);
void     arena_print_stats(const MlArena *a);

#ifdef ML_DEBUG_MEMORY

void ml_mem_report(void);
size_t ml_mem_live_bytes(void);
size_t ml_mem_peak_bytes(void);

#else

static inline void   ml_mem_report(void)          {}
static inline size_t ml_mem_live_bytes(void)       { return 0; }
static inline size_t ml_mem_peak_bytes(void)       { return 0; }
#endif 

#define ML_NEW(T)           ((T*)ml_calloc(1, sizeof(T)))

#define ML_NEW_ARRAY(T, N)  ((T*)ml_calloc((N), sizeof(T)))

#define ML_ZERO(ptr)        memset((ptr), 0, sizeof(*(ptr)))

#endif 
#include "memory.h"

static void oom_abort(const char *fn, size_t size) {
    fprintf(stderr,
            "[ml-memory] FATAL: %s failed to allocate %zu bytes. "
            "Out of memory.\n", fn, size);
    abort();
}

#ifdef ML_DEBUG_MEMORY

typedef struct {
    void       *ptr;
    size_t      size;
    const char *file;
    int         line;
} AllocRecord;

static AllocRecord s_allocs[ML_DEBUG_MAX_ALLOCS];
static int         s_alloc_count  = 0;
static size_t      s_live_bytes   = 0;
static size_t      s_peak_bytes   = 0;

static void dbg_register(void *ptr, size_t size, const char *file, int line) {
    if (s_alloc_count >= ML_DEBUG_MAX_ALLOCS) {
        fprintf(stderr,
                "[ml-memory] WARNING: debug tracker full (%d slots). "
                "Increase ML_DEBUG_MAX_ALLOCS.\n", ML_DEBUG_MAX_ALLOCS);
        return;
    }
    s_allocs[s_alloc_count++] = (AllocRecord){ ptr, size, file, line };
    s_live_bytes += size;
    if (s_live_bytes > s_peak_bytes) s_peak_bytes = s_live_bytes;
}


static size_t dbg_unregister(void *ptr) {
    for (int i = 0; i < s_alloc_count; i++) {
        if (s_allocs[i].ptr == ptr) {
            size_t sz = s_allocs[i].size;
            s_live_bytes -= sz;

            s_allocs[i] = s_allocs[--s_alloc_count];
            return sz;
        }
    }
    fprintf(stderr,
            "[ml-memory] WARNING: ml_free(%p) — pointer not tracked "
            "(double-free or allocated outside ml_malloc?)\n", ptr);
    return 0;
}

void ml_mem_report(void) {
    printf("\n╔══ ml-memory leak report ══════════════════════════════╗\n");
    if (s_alloc_count == 0) {
        printf("║  No leaks detected. All allocations freed.            ║\n");
    } else {
        printf("║  %d live allocation(s), %zu bytes outstanding:\n",
               s_alloc_count, s_live_bytes);
        for (int i = 0; i < s_alloc_count; i++) {
            printf("║   [%3d] %p  %6zu B  %s:%d\n",
                   i,
                   s_allocs[i].ptr,
                   s_allocs[i].size,
                   s_allocs[i].file ? s_allocs[i].file : "?",
                   s_allocs[i].line);
        }
    }
    printf("║  Peak usage: %zu bytes\n", s_peak_bytes);
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
}

size_t ml_mem_live_bytes(void) { return s_live_bytes; }
size_t ml_mem_peak_bytes(void) { return s_peak_bytes; }

/* Internal versions that carry source location. */
static void* _ml_malloc_dbg(size_t size, const char *file, int line) {
    void *p = malloc(size);
    if (!p) oom_abort("ml_malloc", size);
    dbg_register(p, size, file, line);
    return p;
}

static void* _ml_calloc_dbg(size_t n, size_t size, const char *file, int line) {
    void *p = calloc(n, size);
    if (!p) oom_abort("ml_calloc", n * size);
    dbg_register(p, n * size, file, line);
    return p;
}

static void* _ml_realloc_dbg(void *ptr, size_t new_size,
                               const char *file, int line) {
    if (ptr) dbg_unregister(ptr);
    void *p = realloc(ptr, new_size);
    if (!p) oom_abort("ml_realloc", new_size);
    dbg_register(p, new_size, file, line);
    return p;
}


#define ml_malloc(sz)           _ml_malloc_dbg((sz),  __FILE__, __LINE__)
#define ml_calloc(n, sz)        _ml_calloc_dbg((n),(sz), __FILE__, __LINE__)
#define ml_realloc(p, sz)       _ml_realloc_dbg((p),(sz), __FILE__, __LINE__)

#endif 
#ifndef ML_DEBUG_MEMORY  

void* ml_malloc(size_t size) {
    void *p = malloc(size);
    if (!p) oom_abort("ml_malloc", size);
    return p;
}

void* ml_calloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) oom_abort("ml_calloc", n * size);
    return p;
}

void* ml_realloc(void *ptr, size_t new_size) {
    void *p = realloc(ptr, new_size);
    if (!p) oom_abort("ml_realloc", new_size);
    return p;
}

#endif 

void ml_free(void *ptr) {
    if (!ptr) return;
#ifdef ML_DEBUG_MEMORY
    dbg_unregister(ptr);
#endif
    free(ptr);
}

char* ml_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *copy = (char*)ml_malloc(len);
    memcpy(copy, s, len);
    return copy;
}

static size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

MlArena* arena_create(size_t capacity) {
    if (capacity == 0) capacity = ML_ARENA_DEFAULT_CAPACITY;

    MlArena *a = (MlArena*)malloc(sizeof(MlArena));
    if (!a) return NULL;

    a->buf  = (unsigned char*)malloc(capacity);
    if (!a->buf) { free(a); return NULL; }

    a->cap  = capacity;
    a->used = 0;
    a->peak = 0;
    return a;
}

void arena_free(MlArena *a) {
    if (a) {
        free(a->buf);
        free(a);
    }
}

void* arena_alloc(MlArena *a, size_t size) {
    if (!a || size == 0) return NULL;

    size_t aligned = align_up(size, 8);
    if (a->used + aligned > a->cap) {
        fprintf(stderr,
                "[ml-arena] exhausted: requested %zu B, "
                "available %zu / %zu B\n",
                aligned, a->cap - a->used, a->cap);
        return NULL;
    }

    void *ptr = a->buf + a->used;
    a->used  += aligned;
    if (a->used > a->peak) a->peak = a->used;
    return ptr;
}

void* arena_calloc(MlArena *a, size_t n, size_t size) {
    void *p = arena_alloc(a, n * size);
    if (p) memset(p, 0, n * size);
    return p;
}

void arena_reset(MlArena *a) {
    if (a) a->used = 0;
}

void arena_print_stats(const MlArena *a) {
    if (!a) { printf("[ml-arena] NULL\n"); return; }

    double used_mb  = a->used  / (1024.0 * 1024.0);
    double peak_mb  = a->peak  / (1024.0 * 1024.0);
    double cap_mb   = a->cap   / (1024.0 * 1024.0);
    double pct      = a->cap > 0 ? 100.0 * a->used / a->cap : 0.0;

    printf("┌── MlArena stats ─────────────────────────────────┐\n");
    printf("│  Capacity : %8.2f MB (%zu bytes)\n",   cap_mb,  a->cap);
    printf("│  Used     : %8.2f MB (%zu bytes, %.1f%%)\n",
           used_mb, a->used, pct);
    printf("│  Peak     : %8.2f MB (%zu bytes)\n",   peak_mb, a->peak);
    printf("│  Free     : %8.2f MB (%zu bytes)\n",
           (cap_mb - used_mb), a->cap - a->used);
    printf("└──────────────────────────────────────────────────┘\n");
}
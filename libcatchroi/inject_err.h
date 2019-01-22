#ifndef INJECT_ERR_J
#define INJECT_ERR_J

#ifndef DISABLE_ERROR_INJECTION

void __attribute__((constructor)) inject_parse_env();

void inject_start();
void inject_stop();

void register_target_region(int, void*, size_t);


#else // DISABLE_ERROR_INJECTION

static inline void inject_start() {}
static inline void inject_stop() {}

static inline void register_target_region(int __attribute__((unused)) c, void __attribute__((unused)) *p, size_t __attribute__((unused)) s) {}

#endif

#endif // INJECT_ERR_J


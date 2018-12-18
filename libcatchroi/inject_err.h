#ifndef INJECT_ERR_J
#define INJECT_ERR_J

void __attribute__((constructor)) inject_parse_env();

void inject_start();
void inject_stop();

void register_target_region(int, void*, size_t);

#endif // INJECT_ERR_J


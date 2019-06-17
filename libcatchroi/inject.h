#ifndef INJECT_ERR_J
#define INJECT_ERR_J

void __attribute__((constructor)) inject_parse_env();

void inject_start();
void inject_stop();

void register_target_region(int, void*, size_t);
void* inject_error(void* ignore);
void setup_child_perfs(int __attribute__((unused)) signo);
void broadcast_sigalrm(int do_here);

#endif // INJECT_ERR_J


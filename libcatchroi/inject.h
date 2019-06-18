#ifndef INJECT_ERR_J
#define INJECT_ERR_J

void __attribute__((constructor)) inject_parse_env();

void inject_start();
void inject_stop();

void potential_target_region(int, void*, size_t);
void* inject_error(void* ignore);
void handle_child_perfs(int __attribute__((unused)) signo, siginfo_t *siginfo, void __attribute__((unused)) *ctx);
void broadcast_sigalrm(int do_here, int action);

#endif // INJECT_ERR_J


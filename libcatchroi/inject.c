#define _GNU_SOURCE
#include <ctype.h>
#include <dlfcn.h>
#include <err.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "catchroi.h"

#define safe_err(code, myerrmsg) do { \
	const char *errnomsg = strerror(errno); \
	write(STDERR_FILENO, myerrmsg, strlen(myerrmsg)); \
	write(STDERR_FILENO, ": ", 2); \
	write(STDERR_FILENO, errnomsg, strlen(errnomsg)); \
	write(STDERR_FILENO, "\n", 1); \
	_exit(code); \
} while(0)



#define OPTPARSE_IMPLEMENTATION
#include "optparse.h"

#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/syscall.h>

#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>


const int clockid = CLOCK_MONOTONIC_RAW;

enum { WATCHPOINT_DONE = 0, WATCHPOINT_SETUP = 1, WATCHPOINT_GO = 2 };

static inline uint64_t getns()
{
	struct timespec get_time;
	clock_gettime(clockid, &get_time);

	return get_time.tv_sec * 1000000000 + get_time.tv_nsec;
}


#ifdef __x86_64__
# ifdef HAVE_XED
#  include "xed/xed-interface.h"
# endif

const char * const reg_names[] = {
	"AX", "BX", "CX", "DX",
	"SI", "DI", "BP", "SP",
	"IP", "FLAGS", "CS", "SS",
	"DS", "ES", "FS", "GS", // <- those 4 can not be sampled for some reason?
	"R8", "R9", "R10", "R11",
	"R12", "R13", "R14", "R15"
};
#define REGS 0xff0fff
#define NREGS 20
#endif

#ifdef __powerpc64__
const char * const reg_names[] = {
	"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7",
	"R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15",
	"R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23",
	"R24", "R25", "R26", "R27", "R28", "R29", "R30", "R31",
	"NIP", "MSR", "ORIG_R3", "CTR",
	"LINK", "XER", "CCR", "SOFTE",
	"TRAP", "DAR", "DSISR", "SIER",
	"MMCRA" // SIER and MMCRA can not be sampled
};
#define REGS 0x07ffffffffffL
#define NREGS 43
#endif

#ifdef REGS
_Static_assert(__builtin_popcountl(REGS) == NREGS, "register count does not match register mask");
_Static_assert(NREGS < sizeof(reg_names), "too many registers in mask");
#else
#error "Undefined registers names, mask (and weight of mask) for this architecture: see arch/ARCH/include/uapi/asm/perf_regs.h"
#endif



#include "inject.h"

extern int roi_progress();


enum { NONE = 0, FLIP, PUT, DUE };

typedef struct _err
{
	union { int64_t mask; double mask_as_double; };
	volatile int64_t *pos;
	int64_t region, page, tasks_finished;
	uint64_t inject_time, start_time, pre_inject_time, real_inject_time, end_time;
	int n_bits;
	char type, undo, early;
	_Atomic int inj;
	unsigned nthreads, mmap_size;
	intptr_t buf;
	struct perf_event_attr pe;
	pthread_t injector_thread;
} err_t;

__thread int perf_fd = 0;
__thread struct perf_event_mmap_page *local_map = NULL;
static _Atomic unsigned handlers_called = 0;


typedef struct _sample {
	struct perf_event_header header;
	intptr_t ip;
	uint32_t pid, tid;
	uint64_t time, addr;
	uint32_t cpu, _reserved;
	union perf_mem_data_src data_src;
	uint64_t abi, regs[NREGS];
} sample_t;


static err_t *error = NULL;

pthread_t threads[1024], *next_thread = threads + 0;


static inline int64_t pick_bits(const int n_bits)
{
	int64_t flip_bits = 0LL;

	/* Pick distinct bits in [0,63] to flip, simultaneously.
	 * Store them in sorted order, to remember which are already flipped. */
	int i, j, b, distinct_bits[n_bits];
	for (i = 0; i < n_bits; i++)
	{
		/* new bit to flip, as the b-th _unflipped_ bit */
		b = ((double)rand() / RAND_MAX) * (64 - i);

		for (j = i - 1; j >= 0; --j)
		{
			if (distinct_bits[j] >= b + j + 1)
				distinct_bits[j + 1] = distinct_bits[j];
			else
				break;
		}

		distinct_bits[j + 1] = b + j + 1;
		flip_bits ^= 1LL << (b + j + 1);
	}

	return flip_bits;
}


void inject_parse_env()
{
	char *argstr = getenv("INJECT");

	if (argstr == NULL)
		return;

	int seed = 0;
	err_t inject = {.pos = NULL, .type = NONE, .inject_time = 0, .page = -1, .region = -1, .inj = 0};

	// tokenize arguments
	char *argv[128] = {NULL}, *strtok_ctx = NULL;
	argv[0] = strtok_r(argstr, " \t\n\v\f\r", &strtok_ctx);
	for (int i = 1; i < 127; i++)
		if ((argv[i] = strtok_r(NULL, " \t\n\v\f\r", &strtok_ctx)) == NULL)
			break;

	// Standalone option parsing from optparse, because getopt it interferes with programs calling it from main.
	struct optparse_long long_options[] =
	{
		{"n_bits",  'n', OPTPARSE_REQUIRED},
		{"vector",  'v', OPTPARSE_REQUIRED},
		{"page",    'a', OPTPARSE_REQUIRED},
		{"mtbf",    'm', OPTPARSE_REQUIRED},
		{"seed",    's', OPTPARSE_REQUIRED},
		{"put",     'p', OPTPARSE_REQUIRED},
		{"due",     'd', OPTPARSE_NONE},
		{"undo",    'u', OPTPARSE_NONE},
	};

	struct optparse options;
	optparse_init(&options, argv);
    options.optind = 0; // no program name in argv0

	for (int read_all_options = 0; !read_all_options; )
		// NB. options n and p should be mutually exclusive
		switch (optparse_long(&options, long_options, NULL))
		{
		case 'n':
			inject.type        = FLIP;
			inject.n_bits      = atoi(options.optarg);
			break;
		case 'v':
			inject.region      = atoi(options.optarg);
			break;
		case 'a':
			inject.page        = strtoll(options.optarg, NULL, 0);
			break;
		case 'm':
			inject.inject_time = strtoull(options.optarg, NULL, 0);
			break;
		case 's':
			seed               = atoi(options.optarg);
			break;
		case 'u':
			inject.undo        = 1;
			break;
		case 'p':
			inject.type        = PUT;
			inject.mask        = strtoll(options.optarg, NULL, 0);
			break;
		case 'd':
			inject.type        = DUE;
			break;
		case -1:
			read_all_options   = 1;
		}

	if (inject.type == NONE)
		return;
	else if ((inject.page < 0 && inject.region < 0) || inject.inject_time == 0)
		err(1, "Wrong parameters");

	// seed == 0 to get a different seed every time
	srand(seed ? seed : (int)getns());

	inject.inject_time = llround(inject.inject_time * ((double)rand() / (double)RAND_MAX));

	if (inject.type == FLIP)
		inject.mask = pick_bits(inject.n_bits);

	error = memcpy(malloc(sizeof(err_t)), &inject, sizeof(err_t));
	register_mem_region_callback(potential_target_region);

	if (inject.type == DUE)
	{
		struct sigaction alarm = (struct sigaction){.sa_sigaction = handle_child_perfs, .sa_flags = SA_SIGINFO};
		sigemptyset(&alarm.sa_mask);
		sigaddset(&alarm.sa_mask, SIGALRM);
		if (sigaction(SIGALRM, &alarm, NULL) != 0)
			err(-1, "cannot setup SIGALRM handler");
		error->mmap_size = 2 * sysconf(_SC_PAGESIZE);
	}

	*next_thread++ = pthread_self();
}


// from x in a uniform distribution between 0 and 1, get y according to weibull distribution
double weibull(const double lambda, const double k, const double x)
{
	double y, inv_k = 1 / k;
	y = - log1p(-x); // - log(1 - x)
	y = pow(y, inv_k);
	y *= lambda; // where lambda ~ mean time between faults

	return y;
}

// for x uniform between 0 and 1, return -lambda * log(1 - x)
double exponential(const double lambda, const double x)
{
	double y = - log1p(-x); // - log(1 - x)
	y *= lambda;

	return y;
}

static inline
void sleep_ns(uint64_t ns)
{
	struct timespec next_sim_fault, remainder;

	next_sim_fault.tv_sec  = ns / 1000000000ULL;
	next_sim_fault.tv_nsec = ns % 1000000000ULL;

	if (nanosleep(&next_sim_fault, &remainder) != 0)
		fprintf(stderr, "Nanosleep skipped %d.%09d of %d.%09d sleeping time\n",
		        (int)remainder.tv_sec, (int)remainder.tv_nsec, (int)next_sim_fault.tv_sec, (int)next_sim_fault.tv_nsec);
}


void* inject_error(void* ignore)
{
	(void)ignore;

	// default cancellability state + nanosleep is a cancellation point
	sleep_ns(error->inject_time);
	error->pre_inject_time = getns();

	int64_t *ptr = (int64_t*)error->pos;
	if (ptr == NULL)
		error->early++;
	else if (error->type == FLIP)
		*ptr ^= error->mask;
	else if (error->type == PUT)
	{
		int64_t get = *ptr;
		*ptr = error->mask;
		error->mask = get;
	}
	else if (error->type == DUE)
		broadcast_sigalrm(WATCHPOINT_GO);
	else if (error->type != NONE)
		err(-1, "Unrecognised error type");

	error->inj++;
	error->real_inject_time = getns();
	error->tasks_finished = roi_progress();

	return NULL;
}


void handle_child_perfs(int __attribute__((unused)) signo, siginfo_t *siginfo, void __attribute__((unused)) *ctx)
{
	union sigval data = siginfo->si_value;
	if (data.sival_int == WATCHPOINT_SETUP)
	{
		if (!perf_fd)
		{
			perf_fd = syscall(__NR_perf_event_open, &error->pe, 0, -1, -1, 0);
			if (perf_fd < 0)
				safe_err(1, "failed opening child watchpoint");
		}

		local_map = mmap(NULL, error->mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, perf_fd, 0);
		if (MAP_FAILED == local_map)
			safe_err(1, "failed opening child watchpoint's mmap");

		ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
	}
	else if (data.sival_int == WATCHPOINT_GO)
	{
		ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 1);
	}
	else // if DONE, pass the address of where we want to save the samples
	{
		ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
		memcpy(data.sival_ptr, local_map, error->mmap_size);

		munmap(local_map, error->mmap_size);
		close(perf_fd);

		perf_fd = 0;
		local_map = NULL;
	}

	handlers_called++;
}


void inject_start()
{
	if (error == NULL)
		return;

	if (error->type == DUE)
	{
		error->pe = (struct perf_event_attr){
			.type = PERF_TYPE_BREAKPOINT, .bp_type = HW_BREAKPOINT_RW, .bp_len = HW_BREAKPOINT_LEN_8, .bp_addr = (long long)error->pos,
			.size = sizeof (struct perf_event_attr), .config = 0, .pinned = 1, .exclude_kernel = 1, .exclude_hv = 1,
			.sample_period = 1, .wakeup_events = 1, .precise_ip = 3, .sample_regs_intr = REGS, .use_clockid = 1, .clockid = clockid
		};
		error->pe.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TIME | PERF_SAMPLE_TID | PERF_SAMPLE_CPU | PERF_SAMPLE_ADDR | \
						 PERF_SAMPLE_REGS_INTR | PERF_SAMPLE_DATA_SRC;

		error->nthreads = ((intptr_t)next_thread - (intptr_t)threads) / sizeof(*next_thread);
		error->buf = (intptr_t)aligned_alloc(sysconf(_SC_PAGESIZE), error->nthreads * error->mmap_size);

		errno = 0;
		broadcast_sigalrm(WATCHPOINT_SETUP);
	}

	printf("inject_type:%d inject_mask:%#016lx inject_dbl:%g inject_addr:%p inject_back:%d inject_time:%lu\n",
			error->type, error->mask, error->mask_as_double, (void*)error->pos, error->undo, error->inject_time);
	fflush(stdout);

	if (!pthread_create(&error->injector_thread, NULL, &inject_error, (void*)error) == 0)
		err(errno, "Failed to create the injector thread");

	error->start_time = getns();
}


void inject_stop()
{
	if (error == NULL)
		return;

	error->end_time = getns();
	pthread_join(error->injector_thread, NULL);

	/* Print whether we flipped anything or whether the inject region stopped earlier */
	printf("inject_done:%d end_time:%lu inject_finished_tasks:%ld inject_real_before:%lu inject_real_time:%lu",
				error->inj, error->end_time - error->start_time, error->tasks_finished, error->pre_inject_time - error->start_time, error->real_inject_time - error->start_time);

	if (error->inj > 0 && error->undo)
	{
		if (error->type == FLIP)
			*error->pos ^= error->mask;
		else if (error->type == PUT)
		{
			int64_t tmp = *error->pos;
			*error->pos = error->mask;
			error->mask = tmp;
		}
	}

	if (error->type == DUE)
	{
		broadcast_sigalrm(WATCHPOINT_DONE);

#ifdef __x86_64__
# ifdef HAVE_XED
		// Use XED to decode instructions, in particular find out if it was reading or writing
		xed_tables_init();
# endif
#endif

		for (intptr_t map = error->buf; map != error->buf + error->nthreads * error->mmap_size; map += error->mmap_size)
		{
			struct perf_event_mmap_page *event_map = (struct perf_event_mmap_page*)map;
			printf("\ninject_samples:%llu inject_maxsamples:%llu", (event_map->data_head - event_map->data_tail) / sizeof(sample_t), event_map->data_size / sizeof(sample_t));

			const intptr_t evt_start = (intptr_t)event_map + event_map->data_offset + event_map->data_tail;
			const intptr_t evt_end   = (intptr_t)event_map + event_map->data_offset + event_map->data_head;
			for (intptr_t evtptr = evt_start, sample_size = sizeof(struct perf_event_header); evtptr != evt_end; evtptr += sample_size)
			{
				sample_t *sample = (sample_t*)evtptr;
				sample_size = sample->header.size;

				if (sample->header.type != PERF_RECORD_SAMPLE || sample->header.size != sizeof(*sample))
				{
					warnx("Unexpected sample metadata: type=%u size=%u", sample->header.type, sample->header.size);
					continue;
				}

				// print sample data
				printf("\nsample_precise:%d sample_pc:%#016lx sample_time:%lu sample_tid:%u",
						(sample->header.misc & PERF_RECORD_MISC_EXACT_IP) != 0, sample->ip, sample->time - error->start_time, sample->tid);

				// only print sample meta-data if it does not fit with the expected vluaes
				if ((sample->header.misc & PERF_RECORD_MISC_CPUMODE_MASK) != PERF_RECORD_MISC_USER) {
					printf(" sample_cpu_mode:%d", sample->header.misc & PERF_RECORD_MISC_CPUMODE_MASK);
				}

				if (sample->addr != (uintptr_t)error->pos) {
					printf(" sample_address:%#016lx", sample->addr);
				}

				if (sample->data_src.mem_op != PERF_MEM_OP_NA || sample->data_src.mem_lvl != PERF_MEM_LVL_NA) {
					printf(" sample_datasrc_memop:%x sample_datasrc_memlvl:%x", sample->data_src.mem_op, sample->data_src.mem_lvl);
				}

				if (sample->abi != PERF_SAMPLE_REGS_ABI_64) {
					printf(" sample_regs_abi:%s", sample->abi ? "32b" : "none");
				}

				// print all the registers
				printf(" sample_regs");
				for (size_t reg = 0; reg < NREGS; reg++)
					if (REGS & (1 << reg))
						printf(":%s=%016lx", reg_names[reg], sample->regs[reg]);

				// finally decode
#ifdef __x86_64__
# ifdef HAVE_XED
				xed_decoded_inst_t xedd = {0};
				xed_decoded_inst_zero(&xedd);
				xed_decoded_inst_set_mode(&xedd, XED_MACHINE_MODE_LONG_64, XED_ADDRESS_WIDTH_64b);

				if (xed_decode(&xedd, XED_STATIC_CAST(const xed_uint8_t*, sample->ip), 15) != XED_ERROR_NONE)
					printf(" xed_decode failed");

				else
				{
					size_t memops = xed_decoded_inst_number_of_memory_operands(&xedd);
					printf(" sample_memops:%lu", memops);

					for (unsigned m = 0; m < memops; m++)
					{
						printf(" memop%u_read:%d memop%u_write:%d memop%u_writeonly:%d",
							m, xed_decoded_inst_mem_read(&xedd, 0),
							m, xed_decoded_inst_mem_written(&xedd, 0),
							m, xed_decoded_inst_mem_written_only(&xedd, 0));
					}
				}
# else
				printf(" no instruction decoder");
# endif

#else
# ifdef __powerpc64__
				uint32_t *instr = (uint32_t*)sample->ip;

				uint32_t primary_opcode = (*instr >> 26) & 0x3fU;
				uint32_t st = (primary_opcode & 0x24U) == 0x24U;
				uint32_t ld = (primary_opcode & 0x24U) == 0x20U;

				printf("sample_addr:%p sample_instr:%x primary_opcode:%d memop_read:%u memop_write:%u",
							(void*)instr, *instr, primary_opcode, ld, st);
# else
#  error "Architecture not implemented for decoding instructions"
# endif
#endif
			}
		}
	}
	printf("\n");

	free(error);
	fflush(stdout);
}


void potential_target_region(int id, void *target_ptr, size_t target_size)
{
	if (error == NULL || error->pos != NULL)
		return;
	else if (error->region >= 0)
	{
		if (error->region == id)
		{
			intptr_t flip_word = ((double)rand() / RAND_MAX) * (target_size / sizeof(int64_t));
			error->pos = (int64_t*)target_ptr + flip_word;
		}
	}
	else if (error->page >= 0)
	{
		int64_t page_size = 4096; // sysconf(_SC_PAGESIZE) --  WARNING: hardcoded 4K page size
		const int64_t target_pages = (target_size + page_size - 1) / page_size;

		if (error->page < target_pages)
		{
			intptr_t target_left = (intptr_t)target_size - error->page * page_size;
			if (page_size > target_left)
				page_size = target_left;

			intptr_t flip_word = (error->page + (double)rand() / RAND_MAX) * (page_size / sizeof(int64_t));
			error->pos = (int64_t*)target_ptr + flip_word;
		}

		error->page -= target_pages;
	}
	else
		err(-1, "Can not find an injection target!");
}



#undef pthread_create
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void* (*start)(void*), void *arg)
{
	static int (*real_create)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*) = NULL;
	if (!real_create) real_create = dlsym(RTLD_NEXT, "pthread_create");

	Dl_info info;
	if (!dladdr(*(void**)&start, &info))
		dlerror();

	int rc = real_create(thread, attr, start, arg);

	/* Do not instrument thread used to inject errors or failed pthread_creates */
	if (!rc)
	//		(info.dli_saddr == (void*)inject_error && strstr(info.dli_sname, "os_bootthread") != NULL)
	//			|| (info.dli_fname != NULL && strstr(info.dli_fname, "libnanox") != NULL)
		*next_thread++ = *thread;

	return rc;
}

// Called from orchestrating thread or start/stop perf, not from signal handler.
// Used to get all worker (i.e. non-orchestrator) threads to flip their status to state
void broadcast_sigalrm(int action)
{
	pthread_t self = pthread_self();
	union sigval data = {.sival_int = action};
	int here = -1;

	handlers_called = 0;

	for (unsigned pos = 0; pos < error->nthreads; pos++)
		if (threads[pos] != self)
		{
			if (action == WATCHPOINT_DONE) data.sival_ptr = (void*)(error->buf + error->mmap_size * pos);
			pthread_sigqueue(threads[pos], SIGALRM, data);
		}
		else // don't call SIGALRM on self, just run inline below.
			here = pos;

	if (here >= 0)
	{
		if (action == WATCHPOINT_DONE) data.sival_ptr = (void*)(error->buf + error->mmap_size * here);
		siginfo_t info = {.si_value = data};
		handle_child_perfs(0, &info, NULL);
	}

	while (handlers_called < error->nthreads);
}


void __parsec_roi_begin()
{
	static void (*real_roi_begin)() = NULL;
	if (!real_roi_begin) real_roi_begin = dlsym(RTLD_NEXT, "__parsec_roi_begin");

	real_roi_begin();
	inject_start();
}



void stop_roi(int it)
{
	static void (*real_stop_measure)(int) = NULL;
	if (!real_stop_measure) real_stop_measure = dlsym(RTLD_NEXT, "stop_roi");

	inject_stop();
	real_stop_measure(it);
}


void __parsec_roi_end()
{
	static void (*real_roi_end)() = NULL;
	if (!real_roi_end) real_roi_end = dlsym(RTLD_NEXT, "__parsec_roi_end");

	inject_stop();
	real_roi_end();
}

void start_roi () __attribute__ ((noinline, alias ("__parsec_roi_begin")));
void start_measure () __attribute__ ((noinline, alias ("start_roi")));
void stop_measure (int) __attribute__ ((noinline, alias ("stop_roi")));

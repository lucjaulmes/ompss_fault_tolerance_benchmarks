#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <err.h>
#include <errno.h>
#include <time.h>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/syscall.h>

#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>


const int clockid = CLOCK_MONOTONIC_RAW;

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



#include "inject_err.h"

extern int roi_progress();


enum { NONE = 0, FLIP, PUT, DUE };

typedef struct _err
{
	union { int64_t mask; double mask_as_double; };
	volatile int64_t *pos;
	int64_t region, page, tasks_finished;
	uint64_t inject_time, start_time, real_inject_time, end_time;
	int n_bits, perf_fd;
	char type, undo, early;
	_Atomic int inj;
	struct perf_event_mmap_page *event_map;
	pthread_t injector_thread;
} err_t;


typedef struct _sample {
	struct perf_event_header header;
	intptr_t ip;
	uint32_t pid, tid;
	uint64_t time, addr;
	uint32_t cpu, _reserved;
	uint64_t data_src;
	uint64_t abi, regs[NREGS];
} sample_t;


static err_t *error = NULL;


char *token(char **next)
{
	char *ret = NULL;

	for (; **next; (*next)++)
		if (ret == NULL && !isspace(**next))
			ret = *next;
		else if (ret != NULL && isspace(**next))
		{
			**next = '\0';
			(*next)++;
			break;
		}

	return ret;
}


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
#define required_argument 1
#define no_argument 0
	static struct
	{
		const char *name;
		int has_arg, *flag, val;
	} long_options[] =
	{
		{"n_bits",  required_argument, NULL, 'n'},
		{"vector",  required_argument, NULL, 'v'},
		{"page",    required_argument, NULL, 'a'},
		{"mtbf",    required_argument, NULL, 'm'},
		{"seed",    required_argument, NULL, 's'},
		{"put",     required_argument, NULL, 'p'},
		{"due",     no_argument,	   NULL, 'd'},
		{"undo",    no_argument,       NULL, 'u'},
	};

	char *argstr = getenv("INJECT");

	if (argstr == NULL)
		return;

	int seed = 0;
	err_t inject = {.pos = NULL, .type = NONE, .inject_time = 0, .page = -1, .region = -1, .inj = 0};

	// Home-made option parsing, like getopts except don't call on it, since it interferes with calling it from main.
	for (char *optarg = NULL, *optnext = argstr, *optstr = token(&optnext); optstr != NULL; optstr = token(&optnext), optarg = NULL)
	{
		int optlen = strlen(optstr);
		int tok_opts[optlen - 1], n_tok_opts = 0;

		// Parse a single long option
		if (optlen > 2 && optstr[0] == '-' && optstr[1] == '-')
		{
			for (size_t lo = 0; lo < sizeof(long_options) / sizeof(*long_options); lo++)
				if (strcmp(optstr + 2, long_options[lo].name) == 0)
				{
					*tok_opts = long_options[lo].val;
					n_tok_opts = 1;

					if (long_options[lo].has_arg == required_argument)
						// --long <val>
						optarg = token(&optnext);

						// TODO: --long=<val>

					break;
				}
		}
		// Parse a short option(s)
		else if (optlen > 1 && optstr[0] == '-' && optstr[1] != '-')
			// loop on optstr chars, each is (potentially) a short option
			for (n_tok_opts = 0; n_tok_opts < optlen - 1; )
			{
				tok_opts[n_tok_opts] = 0;
				for (size_t lo = 0; lo < sizeof(long_options) / sizeof(*long_options); lo++)
					if ((int)optstr[1 + n_tok_opts] == long_options[lo].val)
					{
						tok_opts[n_tok_opts] = long_options[lo].val;
						if (long_options[lo].has_arg == required_argument)
						{
							// -o<val>
							if (optlen > n_tok_opts + 2)
								optarg = optstr + n_tok_opts + 2;
							// -o <val>
							else
								optarg = token(&optnext);
							// stop loop on short options
							optlen = n_tok_opts;
						}
						break;
					}

				// unrecognized option => break
				if (!tok_opts[n_tok_opts])
					break;
				else
					n_tok_opts++;
			}
		// unrecognized option or '--': break out of here
		else
			break;

		for (int opt_n = 0, opt = tok_opts[opt_n]; opt_n < n_tok_opts; opt = tok_opts[++opt_n])
		{
			// the usual switch() just as if we called opt = getopt(...), even sets optarg.
			// NB. options n and p should be mutually exclusive
			switch (opt)
			{
			case 'n':
				inject.type   = FLIP;
				inject.n_bits = atoi(optarg);
				break;
			case 'v':
				inject.region = atoi(optarg);
				break;
			case 'a':
				inject.page   = strtoll(optarg, NULL, 0);
				break;
			case 'm':
				inject.inject_time   = strtod(optarg, NULL) * rand() / (double)RAND_MAX;
				break;
			case 's':
				seed          = atoi(optarg);
				break;
			case 'u':
				inject.undo   = 1;
				break;
			case 'p':
				inject.type   = PUT;
				inject.mask   = strtoll(optarg, NULL, 0);
				break;
			case 'd':
				inject.type   = DUE;
			}
		}
	}

	if (inject.type == NONE)
		return;
	else if (inject.page < 0 && inject.region < 0)
		err(1, "Wrong parameters");

	// seed == 0 to get a different seed every time
	srand(seed ? seed : (int)getns());

	if (inject.type == FLIP)
		inject.mask = pick_bits(inject.n_bits);

	error = memcpy(malloc(sizeof(err_t)), &inject, sizeof(err_t));
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


inline void
clflush(volatile void *p)
{
    asm volatile ("clflush (%0)" :: "r"(p));
}


inline uint64_t
rdtsc()
{
    unsigned long a, d;
    asm volatile ("cpuid; rdtsc" : "=a" (a), "=d" (d) :: "ebx", "ecx");
    return a | ((uint64_t)d << 32);
}


void* inject_error(void* ignore)
{
	(void)ignore;

	// default cancellability state + nanosleep is a cancellation point
	sleep_ns(error->inject_time);

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
		ioctl(error->perf_fd, PERF_EVENT_IOC_REFRESH, 1);
	else if (error->type != NONE)
		err(-1, "Unrecognised error type");

	error->inj++;
	error->real_inject_time = getns();
	error->tasks_finished = roi_progress();

	return NULL;
}


void inject_start()
{
	if (error == NULL)
		return;

	if (error->type == DUE)
	{
		struct perf_event_attr pe = {
			.type = PERF_TYPE_BREAKPOINT, .bp_type = HW_BREAKPOINT_RW, .bp_len = HW_BREAKPOINT_LEN_8, .bp_addr = (long long)error->pos,
			.size = sizeof (struct perf_event_attr), .config = 0, .pinned = 1, .exclude_kernel = 1, .exclude_hv = 1,
			.sample_period = 1, .wakeup_events = 1, .precise_ip = 3, .sample_regs_intr = REGS, .use_clockid = 1, .clockid = clockid
		};
		pe.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TIME | PERF_SAMPLE_TID | PERF_SAMPLE_CPU | PERF_SAMPLE_ADDR | \
						 PERF_SAMPLE_REGS_INTR | PERF_SAMPLE_DATA_SRC;

		error->perf_fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
		if (error->perf_fd < 0 || MAP_FAILED == (
				 error->event_map = mmap(NULL, 2 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, error->perf_fd, 0)
		))
			err(1, "failed opening watchpoint");

		ioctl(error->perf_fd, PERF_EVENT_IOC_DISABLE, 0);
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
	char buf[1024];
	size_t len = 0;
	#define safe_sprintf(...) do { len += snprintf(buf + len, sizeof(buf) - len, __VA_ARGS__); } while(0)

	safe_sprintf("inject_done:%d end_time:%lu inject_finished_tasks:%ld inject_real_time:%lu",
				error->inj, error->end_time - error->start_time, error->tasks_finished, error->real_inject_time - error->start_time);

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
		safe_sprintf(" inject_samples:%lu", (size_t)(error->event_map->data_head - error->event_map->data_tail) / sizeof(sample_t));

#ifdef __x86_64__
# ifdef HAVE_XED
		// Use XED to decode instructions, in particular find out if it was reading or writing
		xed_tables_init();

		xed_decoded_inst_t xedd = {0};
		xed_decoded_inst_zero(&xedd);
		xed_decoded_inst_set_mode(&xedd, XED_MACHINE_MODE_LONG_64, XED_ADDRESS_WIDTH_64b);
# endif
#endif

		const intptr_t evt_start = (intptr_t)error->event_map + error->event_map->data_offset + error->event_map->data_tail;
		const intptr_t evt_end   = (intptr_t)error->event_map + error->event_map->data_offset + error->event_map->data_head;
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
			len = 0;
			printf("%s\n", buf);
			safe_sprintf("sample_precise:%d sample_pc:%#016lx sample_time:%lu",
					(sample->header.misc & PERF_RECORD_MISC_EXACT_IP) != 0, sample->ip, sample->time - error->start_time);

			// only print sample meta-data if it does not fit with the expected vluaes
			if ((sample->header.misc & PERF_RECORD_MISC_CPUMODE_MASK) != PERF_RECORD_MISC_USER) {
				safe_sprintf(" sample_cpu_mode:%d", sample->header.misc & PERF_RECORD_MISC_CPUMODE_MASK);
			}

			if (sample->addr != (uintptr_t)error->pos) {
				safe_sprintf(" sample_address:%#016lx", sample->addr);
			}

			if (sample->data_src) { // would be helpful but is usually just 0
				int mem_op  = (sample->data_src >> PERF_MEM_OP_SHIFT);
				int mem_lvl = (sample->data_src >> PERF_MEM_LVL_SHIFT);
				safe_sprintf(" sample_datasrc_memop:%x sample_datasrc_memlvl:%x", mem_op, mem_lvl);
			}

			if (sample->abi != PERF_SAMPLE_REGS_ABI_64) {
				safe_sprintf(" sample_regs_abi:%s", sample->abi ? "32b" : "none");
			}

			// print all the registers
			for (size_t reg = 0; reg < NREGS; reg++)
				if (REGS & (1 << reg))
					printf(" sample_%s:%016lx", reg_names[reg], sample->regs[reg]);

			// finally decode
#ifdef __x86_64__
# ifdef HAVE_XED
			if (xed_decode(&xedd, XED_STATIC_CAST(const xed_uint8_t*, sample->ip), 15) != XED_ERROR_NONE)
				safe_sprintf("xed_decode failed");

			else
			{
				size_t memops = xed_decoded_inst_number_of_memory_operands(&xedd);
				safe_sprintf(" sample_memops:%lu", memops);

				for (unsigned m = 0; m < memops; m++)
				{
					len = strnlen(buf, sizeof(buf));
					safe_sprintf(" memop%u_read:%d memop%u_write:%d memop%u_writeonly:%d",
						m, xed_decoded_inst_mem_read(&xedd, 0),
						m, xed_decoded_inst_mem_written(&xedd, 0),
						m, xed_decoded_inst_mem_written_only(&xedd, 0));
				}
			}
# else
			safe_sprintf("no instruction decoder");
# endif

#else
# ifdef __powerpc64__
			uint32_t *instr = (uint32_t*)sample->ip;

			uint32_t primary_opcode = (*instr >> 26) & 0x3fU;
			uint32_t st = (primary_opcode & 0x24U) == 0x24U;
			uint32_t ld = (primary_opcode & 0x24U) == 0x20U;

			safe_sprintf("sample_addr:%p sample_instr:%x primary_opcode:%d memop_read:%u memop_write:%u",
						(void*)instr, *instr, primary_opcode, ld, st);
# else
#  error "Architecture not implemented for decoding instructions"
# endif
#endif
		}

		munmap(error->event_map, error->event_map->data_offset + error->event_map->data_size);
		close(error->perf_fd);
	}

	printf("%s\n", buf);

	free(error);
	fflush(stdout);
}


void register_target_region(int id, void *target_ptr, size_t target_size)
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

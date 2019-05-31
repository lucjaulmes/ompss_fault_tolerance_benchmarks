#include "instrumentation.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "debug.hpp"

#include <sys/time.h>
#include <thread>
#include <mutex>
#include <iomanip>
#include <sstream>

#include "catchroi.h"

namespace nanos {

class InstrumentationCsvTdgTrace : public Instrumentation {
#ifndef NANOS_INSTRUMENTATION_ENABLED
public:
    // constructor
    InstrumentationCsvTdgTrace()
        : Instrumentation()
    {
    }
    // destructor
    ~InstrumentationCsvTdgTrace() {}

    // low-level instrumentation interface (mandatory functions)
    void initialize(void) {}
    void finalize(void) {}
    void disable(void) {}
    void enable(void) {}
    void addResumeTask(WorkDescriptor& w) {}
    void addSuspendTask(WorkDescriptor& w, bool last) {}
    void addEventList(unsigned int count, Event* events) {}
    void threadStart(BaseThread& thread) {}
    void threadFinish(BaseThread& thread) {}
#else
public:
    // constructor
    InstrumentationCsvTdgTrace()
        : Instrumentation(*new InstrumentationContext())
    {
    }
    // destructor
    ~InstrumentationCsvTdgTrace() {}

    struct dep_info
    {
        void *addr;
        uint64_t size;
        const char * const desc;
    };

    struct wd_info
    {
        std::string description;
        int64_t duration;
        std::vector<struct dep_info> deps;
    };


    const char * const dep_name[6] = {"??", "in", "out", "inout", "commutative", "concurrent"};
    std::map<uint64_t, struct wd_info> wds;
    std::map<uint64_t, struct timeval> wd_start_time;
    static std::mutex maps_mutex_;

    static std::mutex current_wd_mutex_;

    inline struct timeval getus()
    {
        struct timeval t;
        gettimeofday(&t, NULL);
        return t;
    }

    // low-level instrumentation interface (mandatory functions)
    void initialize(void)
    {
        InstrumentationDictionary* iD = getInstrumentationDictionary();

        iD->setDefaultLevel(EVENT_NONE);
        iD->switchEventPrefix("create-wd-id", EVENT_ENABLED);
        iD->switchEventPrefix("create-wd-ptr", EVENT_ENABLED);
        iD->switchEventPrefix("wd-num-deps", EVENT_ENABLED);
        iD->switchEventPrefix("wd-deps-ptr", EVENT_ENABLED);
        iD->switchEventPrefix("user-funct-location", EVENT_ENABLED);

        ensure(iD->getEventKey("wd-num-deps") != 0 && iD->getEventKey("wd-deps-ptr"), "Failed to enable events");

        std::lock_guard<std::mutex> lock_maps(maps_mutex_);
        wds.insert(std::make_pair(1, (struct wd_info){"main", 0}));
    }

    void finalize(void)
    {
        struct timeval now = getus();

        std::lock_guard<std::mutex> lock_maps(maps_mutex_);

        // Ensure main() is considered finished
        auto main_start_time = wd_start_time.find(1);
        if (main_start_time != wd_start_time.end()) {
            wds.at(1).duration += (now.tv_sec - main_start_time->second.tv_sec) * 1000000 + (now.tv_usec - main_start_time->second.tv_usec);
            wd_start_time.erase(main_start_time);
        }

        ensure(wd_start_time.empty(), "Some WDs started but never finished");

        for (const auto &wd: wds) {
            std::cerr << wd.first << ':' << wd.second.description << ':' << wd.second.duration << ':' << wd.second.deps.size();
            for (const auto &dep: wd.second.deps)
                std::cerr << ':' << std::hex << (uintptr_t)dep.addr << std::dec << ':' << dep.size << ':' << dep.desc;
            std::cerr << '\n';
        }
    }

    void disable(void) {}
    void enable(void) {}

    void addResumeTask(WorkDescriptor& w)
    {
        struct timeval timestamp = getus();

        std::lock_guard<std::mutex> lock_maps(maps_mutex_);
        bool inserted = false;
        std::tie(std::ignore, inserted) = wd_start_time.insert(std::make_pair(w.getId(), timestamp));
        ensure(inserted, "WD resumed twice without first being suspended");
    }

    void addSuspendTask(WorkDescriptor& w, bool last)
    {
        struct timeval now = getus();
        uint64_t wd_id = w.getId();

        std::lock_guard<std::mutex> lock_maps(maps_mutex_);
        auto start_time = wd_start_time.find(wd_id);
        ensure(start_time != wd_start_time.end(), "WD suspended without first having started");

        wds.at(wd_id).duration += (now.tv_sec - start_time->second.tv_sec) * 1000000 + (now.tv_usec - start_time->second.tv_usec);
        wd_start_time.erase(start_time);
    }

    void addEventList(unsigned int count, Event* events)
    {
        InstrumentationDictionary* iD = getInstrumentationDictionary();
        static const nanos_event_key_t                         create_wd_id  = iD->getEventKey("create-wd-id");
        static const nanos_event_key_t __attribute__((unused)) create_wd_ptr = iD->getEventKey("create-wd-ptr");
        static const nanos_event_key_t __attribute__((unused)) wd_num_deps   = iD->getEventKey("wd-num-deps");
        static const nanos_event_key_t __attribute__((unused)) wd_deps_ptr   = iD->getEventKey("wd-deps-ptr");
        static const nanos_event_key_t                         wd_id         = iD->getEventKey("wd-id");
        static const nanos_event_key_t                         user_func     = iD->getEventKey("user-funct-location");

        // Get the node corresponding to the wd_id calling this function
        // This node won't exist if the calling wd corresponds to that of the master thread
        struct timeval timestamp = getus();
        static __thread int64_t current_wd_id = 0;

        for (Event* e = events; e != events + count; e++) {
            nanos_event_type_t e_type = e->getType();
            nanos_event_key_t e_key = e->getKey();
            nanos_event_key_t e_val = e->getValue();

            if (e_type == NANOS_POINT && e_key == create_wd_id) {
                // NB: this requires --instrument-enable=wd-num-deps,wd-deps-ptr in NX_ARGS
                ensure(count >= 4 && e[1].getKey() == create_wd_ptr && e[2].getKey() == wd_num_deps && e[3].getKey() == wd_deps_ptr,
                        "Unexpected set of events for task creation");

                int64_t created_wd_id = e_val;
                const char* description = ((WD*)e[1].getValue())->getDescription();
                size_t num_deps = e[2].getValue();
                nanos::DataAccess *deps = (nanos::DataAccess*)e[3].getValue();

                struct wd_info wd_data({std::string(description), 0});

                for (nanos::DataAccess *dep = deps; dep != deps + num_deps; ++dep) {
                    // NB. ignores canRename and supposes contiguous dependencies
                    wd_data.deps.emplace_back((struct dep_info){dep->getDepAddress(), dep->getSize(),
                        dep_name[dep->isCommutative() || dep->isConcurrent() ? 4 + dep->isConcurrent() : 2 * dep->isOutput() + dep->isInput()]
                    });
                }

                std::lock_guard<std::mutex> lock_maps(maps_mutex_);
                bool inserted = false;
                std::tie(std::ignore, inserted) = wds.insert(std::make_pair(created_wd_id, wd_data));
                ensure(inserted, "same WD created twice");
                e += 3;

            } else if (e_key == wd_id && e_type == NANOS_BURST_START) {
                current_wd_id = e_val;

            } else if (e_key == wd_id && e_type == NANOS_BURST_END) {
                ensure(e_val == current_wd_id, "ending WD which was not started");
                current_wd_id = 0;

            } else if (e_key == user_func && e_type == NANOS_BURST_START) {
                task_started(current_wd_id);
                std::lock_guard<std::mutex> lock_maps(maps_mutex_);

                bool inserted = false;
                std::tie(std::ignore, inserted) = wd_start_time.insert(std::make_pair(current_wd_id, timestamp));
                ensure(inserted, "same WD started twice");

            } else if (e_key == user_func && e_type == NANOS_BURST_END) {
                task_ended(current_wd_id);
                std::lock_guard<std::mutex> lock_maps(maps_mutex_);

                auto start_time = wd_start_time.find(current_wd_id);
                ensure(start_time != wd_start_time.end(), "WD stopped without first having started");

                wds.at(start_time->first).duration += (timestamp.tv_sec - start_time->second.tv_sec) * 1000000
                                                    + (timestamp.tv_usec - start_time->second.tv_usec);
                wd_start_time.erase(start_time);
            }
        }
    }
    void threadStart(BaseThread& thread) {}
    void threadFinish(BaseThread& thread) {}
#endif
};

namespace ext {

    class InstrumentationCsvTdgTracePlugin : public Plugin {
    public:
        InstrumentationCsvTdgTracePlugin()
            : Plugin("Instrumentation generating a simple csv trace of tasks and dependencies.", 1)
        {
        }
        ~InstrumentationCsvTdgTracePlugin() {}

        void config(Config& cfg) {}

        void init()
        {
            sys.setInstrumentation(new InstrumentationCsvTdgTrace());
        }
    };

} // namespace ext

#ifdef NANOS_INSTRUMENTATION_ENABLED
std::mutex InstrumentationCsvTdgTrace::maps_mutex_;
//std::mutex InstrumentationCsvTdgTrace::current_wd_mutex_;
#endif

} // namespace nanos

DECLARE_PLUGIN("instrumentation-catch_tasks_deps", nanos::ext::InstrumentationCsvTdgTracePlugin);

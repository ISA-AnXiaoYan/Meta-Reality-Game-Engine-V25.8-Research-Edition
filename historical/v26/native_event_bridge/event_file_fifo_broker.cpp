#include <algorithm>
#include <atomic>
#include <csignal>
#include <cctype>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>
#include <metavision/sdk/base/utils/error_utils.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/file_config_hints.h>

namespace {

constexpr uint16_t kWireVersion = 1;
constexpr uint32_t kFlagHello   = 1u << 0;
constexpr uint32_t kFlagFinal   = 1u << 1;

#pragma pack(push, 1)
struct WireHeader {
    char magic[4];
    uint16_t version;
    uint16_t header_size;
    uint32_t flags;
    uint32_t width;
    uint32_t height;
    uint64_t seq;
    int64_t slice_start_us;
    int64_t slice_end_us;
    int64_t current_time_us;
    uint32_t event_count;
    uint32_t trigger_count;
    uint32_t event_record_size;
    uint32_t trigger_record_size;
    uint32_t payload_crc;
    uint32_t reserved;
};

struct WireEventCD {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int64_t t;
};

struct WireExtTrigger {
    int16_t p;
    int16_t id;
    int64_t t;
};
#pragma pack(pop)

static_assert(sizeof(WireEventCD) == 14, "WireEventCD must match Python dtype itemsize");
static_assert(sizeof(WireExtTrigger) == 12, "WireExtTrigger must match Python dtype itemsize");

std::atomic<bool> g_stop{false};

void signal_handler(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

int64_t steady_us() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

int64_t wall_us() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

std::string trim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
        return !std::isspace(c);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) {
        return !std::isspace(c);
    }).base(), s.end());
    return s;
}

std::vector<std::string> split_csv(const std::string &csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::string json_escape(const std::string &s) {
    std::ostringstream os;
    for (char c : s) {
        switch (c) {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default: os << c; break;
        }
    }
    return os.str();
}

std::string clean_alias(const std::string &alias) {
    std::string out;
    for (char c : alias) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            out.push_back(c);
        }
    }
    return out.empty() ? "cam" : out;
}

std::string dirname_of(const std::string &path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return path.substr(0, 1);
    }
    return path.substr(0, pos);
}

bool ensure_dir(const std::string &path, std::string &err) {
    if (path.empty() || path == ".") {
        return true;
    }
    std::string cur;
    size_t i = 0;
    if (path.size() >= 1 && path[0] == '/') {
        cur = "/";
        i = 1;
    }
    while (i <= path.size()) {
        size_t j = path.find('/', i);
        std::string part = path.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (!part.empty() && part != ".") {
            if (!cur.empty() && cur.back() != '/') {
                cur += "/";
            }
            cur += part;
            struct stat st {};
            if (::stat(cur.c_str(), &st) != 0) {
                if (::mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) {
                    err = cur + ": " + std::strerror(errno);
                    return false;
                }
            } else if (!S_ISDIR(st.st_mode)) {
                err = cur + " exists and is not a directory";
                return false;
            }
        }
        if (j == std::string::npos) {
            break;
        }
        i = j + 1;
    }
    return true;
}

bool ensure_parent_dir(const std::string &path, std::string &err) {
    return ensure_dir(dirname_of(path), err);
}

bool prepare_fifo(const std::string &path, std::string &err) {
    struct stat st {};
    if (::stat(path.c_str(), &st) == 0) {
        if (S_ISFIFO(st.st_mode)) {
            return true;
        }
        if (::unlink(path.c_str()) != 0) {
            err = "unlink existing non-fifo failed: " + std::string(std::strerror(errno));
            return false;
        }
    }
    if (::mkfifo(path.c_str(), 0666) != 0 && errno != EEXIST) {
        err = "mkfifo failed: " + std::string(std::strerror(errno));
        return false;
    }
    return true;
}

bool write_all_fd(int fd, const void *data, size_t n, std::string &err) {
    const auto *p = static_cast<const uint8_t *>(data);
    size_t remain = n;
    while (remain > 0 && !g_stop.load(std::memory_order_relaxed)) {
        const ssize_t w = ::write(fd, p, remain);
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            err = std::string("write failed: ") + std::strerror(errno);
            return false;
        }
        if (w == 0) {
            err = "write returned zero";
            return false;
        }
        p += w;
        remain -= static_cast<size_t>(w);
    }
    return remain == 0;
}

struct Options {
    std::string raw_files_csv;
    std::string aliases_csv;
    std::string output_mode = "fifo";
    std::string out_dir = "sync_ipc/native_hal_fifos";
    std::string fifo_open_mode = "rdwr";
    std::string stats_jsonl = "sync_ipc/event_file_fifo_broker_stats.jsonl";
    std::string replay_timing = "realtime"; // realtime or fast
    int slice_us = 4000;
    int report_ms = 1000;
    bool inspect_only = false;
    bool allow_missing_ext_trigger = false;
    bool time_shift = true;
    bool loop = false;
    size_t max_read_per_op = 4 * 1024 * 1024;
    size_t max_memory = 12 * 1024 * 1024;
};

struct FileSpec {
    std::string alias;
    std::string path;
};

std::vector<FileSpec> parse_raw_files(const Options &opt) {
    std::vector<FileSpec> specs;
    auto raw_items = split_csv(opt.raw_files_csv);
    auto aliases = split_csv(opt.aliases_csv);
    for (size_t i = 0; i < raw_items.size(); ++i) {
        std::string item = raw_items[i];
        std::string alias;
        std::string path;
        auto eq = item.find('=');
        auto colon = item.find(':');
        size_t sep = std::string::npos;
        if (eq != std::string::npos) {
            sep = eq;
        } else if (colon != std::string::npos && colon > 0 && colon < 4) {
            sep = colon;
        }
        if (sep != std::string::npos) {
            alias = trim(item.substr(0, sep));
            path = trim(item.substr(sep + 1));
        } else {
            alias = i < aliases.size() ? aliases[i] : ("cam" + std::to_string(i));
            path = item;
        }
        if (alias.empty()) {
            alias = i < aliases.size() ? aliases[i] : ("cam" + std::to_string(i));
        }
        if (!path.empty()) {
            specs.push_back(FileSpec{alias, path});
        }
    }
    return specs;
}

struct ReplayStats {
    std::string alias;
    std::string raw_file;
    std::string fifo_path;
    std::string last_error;
    uint32_t width = 0;
    uint32_t height = 0;
    std::atomic<bool> opened{false};
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> ext_trigger_decoder_available{false};
    std::atomic<uint64_t> callbacks{0};
    std::atomic<uint64_t> trigger_callbacks{0};
    std::atomic<uint64_t> events_seen{0};
    std::atomic<uint64_t> triggers_seen{0};
    std::atomic<uint64_t> frames_written{0};
    std::atomic<uint64_t> events_written{0};
    std::atomic<uint64_t> triggers_written{0};
    std::atomic<uint64_t> write_errors{0};
    std::atomic<uint64_t> exceptions{0};
    std::atomic<uint64_t> loops_completed{0};

    mutable std::mutex mu;
    int64_t first_ts = -1;
    int64_t last_ts = -1;
    int64_t first_ext_trigger_ts_us = -1;
    int64_t last_ext_trigger_ts_us = -1;
    int64_t max_ext_trigger_gap_us = 0;
    std::vector<int64_t> ext_trigger_gaps_us;
    struct TriggerTrackStats {
        int16_t id = 0;
        int16_t p = 0;
        uint64_t count = 0;
        int64_t first_ts_us = -1;
        int64_t last_ts_us = -1;
        int64_t max_gap_us = 0;
        std::vector<int64_t> gaps_us;
    };
    std::map<std::string, TriggerTrackStats> ext_trigger_tracks;

    void set_error(const std::string &msg) {
        exceptions.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mu);
        last_error = msg;
    }

    void on_ts(int64_t t) {
        std::lock_guard<std::mutex> lk(mu);
        if (first_ts < 0) {
            first_ts = t;
        }
        last_ts = std::max<int64_t>(last_ts, t);
    }

    static int64_t median_copy(std::vector<int64_t> values) {
        if (values.empty()) {
            return -1;
        }
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        return values[values.size() / 2];
    }

    void on_trigger_event(int16_t id, int16_t p, int64_t t) {
        std::lock_guard<std::mutex> lk(mu);
        if (first_ext_trigger_ts_us < 0) {
            first_ext_trigger_ts_us = t;
        }
        if (last_ext_trigger_ts_us >= 0) {
            const int64_t gap = t - last_ext_trigger_ts_us;
            if (gap >= 0) {
                ext_trigger_gaps_us.push_back(gap);
                max_ext_trigger_gap_us = std::max(max_ext_trigger_gap_us, gap);
            }
        }
        last_ext_trigger_ts_us = std::max<int64_t>(last_ext_trigger_ts_us, t);

        const std::string key = std::to_string(id) + ":" + std::to_string(p);
        auto &track = ext_trigger_tracks[key];
        track.id = id;
        track.p = p;
        if (track.count == 0) {
            track.first_ts_us = t;
        }
        if (track.last_ts_us >= 0) {
            const int64_t gap = t - track.last_ts_us;
            if (gap >= 0) {
                track.gaps_us.push_back(gap);
                track.max_gap_us = std::max(track.max_gap_us, gap);
            }
        }
        track.last_ts_us = std::max<int64_t>(track.last_ts_us, t);
        ++track.count;
    }

    int64_t median_gap_us_locked() const {
        if (ext_trigger_gaps_us.empty()) {
            return -1;
        }
        auto gaps = ext_trigger_gaps_us;
        std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
        return gaps[gaps.size() / 2];
    }

    std::string json_snapshot() const {
        std::lock_guard<std::mutex> lk(mu);
        const bool has_triggers = triggers_seen.load(std::memory_order_relaxed) > 0;
        std::ostringstream os;
        os << "{";
        os << "\"alias\":\"" << json_escape(alias) << "\",";
        os << "\"raw_file\":\"" << json_escape(raw_file) << "\",";
        os << "\"fifo_path\":\"" << json_escape(fifo_path) << "\",";
        os << "\"width\":" << width << ",";
        os << "\"height\":" << height << ",";
        os << "\"opened\":" << (opened.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"started\":" << (started.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"finished\":" << (finished.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"ext_trigger_decoder_available\":"
           << (ext_trigger_decoder_available.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"callbacks\":" << callbacks.load(std::memory_order_relaxed) << ",";
        os << "\"trigger_callbacks\":" << trigger_callbacks.load(std::memory_order_relaxed) << ",";
        os << "\"events_seen\":" << events_seen.load(std::memory_order_relaxed) << ",";
        os << "\"triggers_seen\":" << triggers_seen.load(std::memory_order_relaxed) << ",";
        os << "\"has_ext_trigger\":" << (has_triggers ? "true" : "false") << ",";
        os << "\"valid_for_replay_sync\":" << (has_triggers ? "true" : "false") << ",";
        os << "\"frames_written\":" << frames_written.load(std::memory_order_relaxed) << ",";
        os << "\"events_written\":" << events_written.load(std::memory_order_relaxed) << ",";
        os << "\"triggers_written\":" << triggers_written.load(std::memory_order_relaxed) << ",";
        os << "\"write_errors\":" << write_errors.load(std::memory_order_relaxed) << ",";
        os << "\"exceptions\":" << exceptions.load(std::memory_order_relaxed) << ",";
        os << "\"loops_completed\":" << loops_completed.load(std::memory_order_relaxed) << ",";
        os << "\"first_ts_us\":" << first_ts << ",";
        os << "\"last_ts_us\":" << last_ts << ",";
        os << "\"first_ext_trigger_ts_us\":" << first_ext_trigger_ts_us << ",";
        os << "\"last_ext_trigger_ts_us\":" << last_ext_trigger_ts_us << ",";
        os << "\"median_ext_trigger_period_us\":" << median_gap_us_locked() << ",";
        os << "\"max_ext_trigger_gap_us\":" << max_ext_trigger_gap_us << ",";
        os << "\"ext_trigger_tracks\":[";
        bool first_track = true;
        for (const auto &kv : ext_trigger_tracks) {
            if (!first_track) {
                os << ",";
            }
            first_track = false;
            const auto &track = kv.second;
            os << "{";
            os << "\"id\":" << track.id << ",";
            os << "\"p\":" << track.p << ",";
            os << "\"count\":" << track.count << ",";
            os << "\"first_ts_us\":" << track.first_ts_us << ",";
            os << "\"last_ts_us\":" << track.last_ts_us << ",";
            os << "\"median_period_us\":" << median_copy(track.gaps_us) << ",";
            os << "\"max_gap_us\":" << track.max_gap_us;
            os << "}";
        }
        os << "],";
        os << "\"last_error\":\"" << json_escape(last_error) << "\"";
        os << "}";
        return os.str();
    }
};

struct FrameWriter {
    ReplayStats *stats = nullptr;
    bool dry_run = false;
    int fd = -1;

    bool open_for_camera(const Options &opt, const std::string &alias, ReplayStats &st) {
        stats = &st;
        dry_run = opt.inspect_only || opt.output_mode == "dry-run";
        if (dry_run) {
            return true;
        }
        std::string err;
        if (!ensure_dir(opt.out_dir, err)) {
            st.set_error("out_dir: " + err);
            return false;
        }
        const std::string path = opt.out_dir + "/event_fifo_" + clean_alias(alias) + ".evb";
        st.fifo_path = path;
        if (!prepare_fifo(path, err)) {
            st.set_error("prepare_fifo " + path + ": " + err);
            return false;
        }
        int flags = O_WRONLY;
        if (opt.fifo_open_mode == "rdwr") {
            flags = O_RDWR;
        } else if (opt.fifo_open_mode == "nonblock") {
            flags = O_WRONLY | O_NONBLOCK;
        }
        fd = ::open(path.c_str(), flags);
        if (fd < 0) {
            st.set_error("open_fifo " + path + ": " + std::strerror(errno));
            return false;
        }
        return true;
    }

    bool write_frame(uint32_t width, uint32_t height, uint64_t seq, uint32_t flags, int64_t slice_start_us,
                     int64_t slice_end_us, int64_t current_time_us, const std::vector<WireEventCD> &events,
                     const std::vector<WireExtTrigger> &triggers) {
        if (stats) {
            stats->frames_written.fetch_add(1, std::memory_order_relaxed);
            stats->events_written.fetch_add(static_cast<uint64_t>(events.size()), std::memory_order_relaxed);
            stats->triggers_written.fetch_add(static_cast<uint64_t>(triggers.size()), std::memory_order_relaxed);
        }
        if (dry_run) {
            return true;
        }
        if (fd < 0) {
            if (stats) {
                stats->write_errors.fetch_add(1, std::memory_order_relaxed);
                stats->set_error("fifo fd is not open");
            }
            return false;
        }

        WireHeader h {};
        h.magic[0] = 'E';
        h.magic[1] = 'V';
        h.magic[2] = 'B';
        h.magic[3] = '1';
        h.version = kWireVersion;
        h.header_size = static_cast<uint16_t>(sizeof(WireHeader));
        h.flags = flags;
        h.width = width;
        h.height = height;
        h.seq = seq;
        h.slice_start_us = slice_start_us;
        h.slice_end_us = slice_end_us;
        h.current_time_us = current_time_us;
        h.event_count = static_cast<uint32_t>(events.size());
        h.trigger_count = static_cast<uint32_t>(triggers.size());
        h.event_record_size = static_cast<uint32_t>(sizeof(WireEventCD));
        h.trigger_record_size = static_cast<uint32_t>(sizeof(WireExtTrigger));

        std::string err;
        bool ok = write_all_fd(fd, &h, sizeof(h), err);
        if (ok && !events.empty()) {
            ok = write_all_fd(fd, events.data(), events.size() * sizeof(WireEventCD), err);
        }
        if (ok && !triggers.empty()) {
            ok = write_all_fd(fd, triggers.data(), triggers.size() * sizeof(WireExtTrigger), err);
        }
        if (!ok && stats) {
            stats->write_errors.fetch_add(1, std::memory_order_relaxed);
            stats->set_error(err);
        }
        return ok;
    }

    void close() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
};

struct SliceEmitter {
    uint32_t width = 0;
    uint32_t height = 0;
    int slice_us = 4000;
    uint64_t seq = 0;
    int64_t current_idx = -1;
    int64_t time_offset_us = 0;
    FrameWriter *writer = nullptr;
    std::vector<WireEventCD> current_events;
    std::map<int64_t, std::vector<WireExtTrigger>> pending_triggers;

    bool emit_hello() {
        const std::vector<WireEventCD> empty_events;
        const std::vector<WireExtTrigger> empty_triggers;
        return writer && writer->write_frame(width, height, seq++, kFlagHello, 0, 0, 0, empty_events, empty_triggers);
    }

    bool emit_idx(int64_t idx) {
        auto trig_it = pending_triggers.find(idx);
        std::vector<WireExtTrigger> triggers;
        if (trig_it != pending_triggers.end()) {
            triggers.swap(trig_it->second);
            pending_triggers.erase(trig_it);
        }
        const int64_t start_us = idx * static_cast<int64_t>(slice_us);
        const int64_t end_us = start_us + static_cast<int64_t>(slice_us);
        const bool ok = writer && writer->write_frame(width, height, seq++, 0, start_us, end_us, end_us,
                                                      current_events, triggers);
        current_events.clear();
        return ok;
    }

    bool advance_to(int64_t idx) {
        if (current_idx < 0) {
            current_idx = idx;
            return true;
        }
        while (current_idx < idx && !g_stop.load(std::memory_order_relaxed)) {
            if (!emit_idx(current_idx)) {
                return false;
            }
            ++current_idx;
        }
        return true;
    }

    bool on_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (const auto *ev = begin; ev != end; ++ev) {
            const int64_t shifted_t = static_cast<int64_t>(ev->t) + time_offset_us;
            const int64_t idx = static_cast<int64_t>(shifted_t / slice_us);
            if (!advance_to(idx)) {
                return false;
            }
            current_events.push_back(WireEventCD{
                static_cast<uint16_t>(ev->x),
                static_cast<uint16_t>(ev->y),
                static_cast<int16_t>(ev->p),
                shifted_t,
            });
        }
        return true;
    }

    bool on_triggers(const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
        for (const auto *ev = begin; ev != end; ++ev) {
            const int64_t shifted_t = static_cast<int64_t>(ev->t) + time_offset_us;
            const int64_t idx = static_cast<int64_t>(shifted_t / slice_us);
            WireExtTrigger trigger {
                static_cast<int16_t>(ev->p),
                static_cast<int16_t>(ev->id),
                shifted_t,
            };
            if (current_idx >= 0 && idx < current_idx) {
                const std::vector<WireEventCD> empty_events;
                const std::vector<WireExtTrigger> triggers{trigger};
                const int64_t start_us = idx * static_cast<int64_t>(slice_us);
                if (!writer ||
                    !writer->write_frame(width, height, seq++, 0, start_us, start_us + slice_us,
                                         start_us + slice_us, empty_events, triggers)) {
                    return false;
                }
                continue;
            }
            pending_triggers[idx].push_back(trigger);
        }
        return true;
    }

    void finalize(bool emit_final = true) {
        if (current_idx >= 0) {
            emit_idx(current_idx);
            ++current_idx;
        }
        for (auto &kv : pending_triggers) {
            current_events.clear();
            auto triggers = std::move(kv.second);
            const int64_t start_us = kv.first * static_cast<int64_t>(slice_us);
            if (writer) {
                writer->write_frame(width, height, seq++, 0, start_us, start_us + slice_us, start_us + slice_us,
                                    current_events, triggers);
            }
        }
        pending_triggers.clear();
        current_events.clear();
        if (emit_final) {
            const std::vector<WireEventCD> empty_events;
            const std::vector<WireExtTrigger> empty_triggers;
            if (writer) {
                writer->write_frame(width, height, seq++, kFlagFinal, 0, 0, 0, empty_events, empty_triggers);
            }
        }
    }

    void advance_loop_offset() {
        const int64_t next_idx = current_idx < 0 ? 0 : current_idx + 1;
        time_offset_us = next_idx * static_cast<int64_t>(slice_us);
    }
};

struct ReplayRuntime {
    FileSpec spec;
    Metavision::Camera camera;
    ReplayStats stats;
    FrameWriter writer;
    SliceEmitter emitter;
    std::mutex emitter_mu;
};

Options parse_args(int argc, char **argv) {
    Options opt;
    auto value = [&](int &i) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + argv[i]);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--raw-files") {
            opt.raw_files_csv = value(i);
        } else if (arg == "--aliases") {
            opt.aliases_csv = value(i);
        } else if (arg == "--output-mode") {
            opt.output_mode = value(i);
        } else if (arg == "--out-dir") {
            opt.out_dir = value(i);
        } else if (arg == "--fifo-open-mode") {
            opt.fifo_open_mode = value(i);
        } else if (arg == "--stats-jsonl") {
            opt.stats_jsonl = value(i);
        } else if (arg == "--slice-us") {
            opt.slice_us = std::stoi(value(i));
        } else if (arg == "--report-ms") {
            opt.report_ms = std::stoi(value(i));
        } else if (arg == "--replay-timing") {
            opt.replay_timing = value(i);
        } else if (arg == "--inspect-only") {
            opt.inspect_only = true;
            opt.output_mode = "dry-run";
        } else if (arg == "--allow-missing-ext-trigger") {
            opt.allow_missing_ext_trigger = true;
        } else if (arg == "--loop") {
            opt.loop = true;
        } else if (arg == "--no-time-shift") {
            opt.time_shift = false;
        } else if (arg == "--max-read-per-op") {
            opt.max_read_per_op = static_cast<size_t>(std::stoull(value(i)));
        } else if (arg == "--max-memory") {
            opt.max_memory = static_cast<size_t>(std::stoull(value(i)));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: event_file_fifo_broker --raw-files A=file.raw,B=file.raw "
                         "[--output-mode fifo|dry-run --out-dir sync_ipc/native_hal_fifos] "
                         "[--replay-timing realtime|fast --loop --inspect-only --allow-missing-ext-trigger]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.raw_files_csv.empty()) {
        throw std::runtime_error("--raw-files is required");
    }
    if (opt.output_mode != "fifo" && opt.output_mode != "dry-run") {
        throw std::runtime_error("--output-mode must be fifo or dry-run");
    }
    if (opt.fifo_open_mode != "rdwr" && opt.fifo_open_mode != "blocking" && opt.fifo_open_mode != "nonblock") {
        throw std::runtime_error("--fifo-open-mode must be rdwr, blocking, or nonblock");
    }
    if (opt.replay_timing != "realtime" && opt.replay_timing != "fast") {
        throw std::runtime_error("--replay-timing must be realtime or fast");
    }
    opt.slice_us = std::max(1, opt.slice_us);
    opt.report_ms = std::max(100, opt.report_ms);
    return opt;
}

std::string aggregate_json(const std::vector<std::unique_ptr<ReplayRuntime>> &runtimes, uint64_t tick,
                           const Options &opt) {
    uint64_t events = 0, triggers = 0, frames = 0, errors = 0, opened = 0, finished = 0;
    for (const auto &rt : runtimes) {
        const auto &st = rt->stats;
        events += st.events_seen.load(std::memory_order_relaxed);
        triggers += st.triggers_seen.load(std::memory_order_relaxed);
        frames += st.frames_written.load(std::memory_order_relaxed);
        errors += st.write_errors.load(std::memory_order_relaxed) + st.exceptions.load(std::memory_order_relaxed);
        opened += st.opened.load(std::memory_order_relaxed) ? 1 : 0;
        finished += st.finished.load(std::memory_order_relaxed) ? 1 : 0;
    }
    std::ostringstream os;
    os << "{";
    os << "\"schema\":\"event_file_fifo_broker_stats.v1\",";
    os << "\"source_mode\":\"file_replay\",";
    os << "\"sync_policy\":\"raw_external_trigger_first\",";
    os << "\"tick\":" << tick << ",";
    os << "\"published_wall_us\":" << wall_us() << ",";
    os << "\"inspect_only\":" << (opt.inspect_only ? "true" : "false") << ",";
    os << "\"output_mode\":\"" << json_escape(opt.output_mode) << "\",";
    os << "\"replay_timing\":\"" << json_escape(opt.replay_timing) << "\",";
    os << "\"time_shift\":" << (opt.time_shift ? "true" : "false") << ",";
    os << "\"loop\":" << (opt.loop ? "true" : "false") << ",";
    os << "\"slice_us\":" << opt.slice_us << ",";
    os << "\"allow_missing_ext_trigger\":" << (opt.allow_missing_ext_trigger ? "true" : "false") << ",";
    os << "\"cameras_requested\":" << runtimes.size() << ",";
    os << "\"cameras_opened\":" << opened << ",";
    os << "\"cameras_finished\":" << finished << ",";
    os << "\"events_seen\":" << events << ",";
    os << "\"triggers_seen\":" << triggers << ",";
    os << "\"frames_written\":" << frames << ",";
    os << "\"errors\":" << errors << ",";
    os << "\"cameras\":[";
    for (size_t i = 0; i < runtimes.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << runtimes[i]->stats.json_snapshot();
    }
    os << "]}";
    return os.str();
}

bool configure_camera(ReplayRuntime &rt, const Options &opt) {
    try {
        Metavision::FileConfigHints hints;
        hints.real_time_playback(opt.replay_timing == "realtime");
        hints.time_shift(opt.time_shift);
        hints.max_read_per_op(opt.max_read_per_op);
        hints.max_memory(opt.max_memory);
        rt.camera = Metavision::Camera::from_file(rt.spec.path, hints);
        rt.stats.opened.store(true, std::memory_order_relaxed);
        rt.stats.started.store(false, std::memory_order_relaxed);
        rt.stats.finished.store(false, std::memory_order_relaxed);
        try {
            rt.stats.width = static_cast<uint32_t>(std::max(0, rt.camera.geometry().width()));
            rt.stats.height = static_cast<uint32_t>(std::max(0, rt.camera.geometry().height()));
        } catch (const std::exception &) {
            rt.stats.width = 1280;
            rt.stats.height = 720;
        }
        rt.emitter.width = rt.stats.width;
        rt.emitter.height = rt.stats.height;
        rt.emitter.slice_us = opt.slice_us;

        rt.camera.cd().add_callback([&rt](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            const uint64_t n = static_cast<uint64_t>(std::distance(begin, end));
            rt.stats.callbacks.fetch_add(1, std::memory_order_relaxed);
            rt.stats.events_seen.fetch_add(n, std::memory_order_relaxed);
            if (n == 0) {
                return;
            }
            const int64_t offset = rt.emitter.time_offset_us;
            rt.stats.on_ts(static_cast<int64_t>(begin->t) + offset);
            rt.stats.on_ts(static_cast<int64_t>((end - 1)->t) + offset);
            std::lock_guard<std::mutex> lk(rt.emitter_mu);
            if (!rt.emitter.on_events(begin, end)) {
                rt.stats.set_error("emitter.on_events failed");
                g_stop.store(true, std::memory_order_relaxed);
            }
        });

        try {
            rt.camera.ext_trigger().add_callback(
                [&rt](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                    const uint64_t n = static_cast<uint64_t>(std::distance(begin, end));
                    rt.stats.trigger_callbacks.fetch_add(1, std::memory_order_relaxed);
                    rt.stats.triggers_seen.fetch_add(n, std::memory_order_relaxed);
                    if (n == 0) {
                        return;
                    }
                    const int64_t offset = rt.emitter.time_offset_us;
                    for (auto *ev = begin; ev != end; ++ev) {
                        rt.stats.on_trigger_event(
                            static_cast<int16_t>(ev->id),
                            static_cast<int16_t>(ev->p),
                            static_cast<int64_t>(ev->t) + offset
                        );
                    }
                    std::lock_guard<std::mutex> lk(rt.emitter_mu);
                    if (!rt.emitter.on_triggers(begin, end)) {
                        rt.stats.set_error("emitter.on_triggers failed");
                        g_stop.store(true, std::memory_order_relaxed);
                    }
                });
            rt.stats.ext_trigger_decoder_available.store(true, std::memory_order_relaxed);
        } catch (const std::exception &e) {
            rt.stats.ext_trigger_decoder_available.store(false, std::memory_order_relaxed);
            rt.stats.set_error(std::string("ext_trigger callback unavailable: ") + e.what());
        }

        rt.camera.add_runtime_error_callback([&rt](const Metavision::CameraException &e) {
            rt.stats.set_error(std::string("runtime_error: ") + e.what());
            g_stop.store(true, std::memory_order_relaxed);
        });
    } catch (const Metavision::BaseException &e) {
        rt.stats.set_error(std::string("Metavision BaseException: ") + e.what());
        return false;
    } catch (const std::exception &e) {
        rt.stats.set_error(std::string("exception: ") + e.what());
        return false;
    }
    return true;
}

bool init_runtime(ReplayRuntime &rt, const Options &opt) {
    rt.stats.alias = rt.spec.alias;
    rt.stats.raw_file = rt.spec.path;
    rt.emitter.slice_us = opt.slice_us;
    rt.emitter.writer = &rt.writer;
    if (!rt.writer.open_for_camera(opt, rt.spec.alias, rt.stats)) {
        return false;
    }
    return configure_camera(rt, opt);
}

} // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        const Options opt = parse_args(argc, argv);
        const auto specs = parse_raw_files(opt);
        if (specs.empty()) {
            throw std::runtime_error("no raw files after parsing --raw-files");
        }

        std::string dir_err;
        if (!opt.inspect_only && opt.output_mode == "fifo" && !ensure_dir(opt.out_dir, dir_err)) {
            throw std::runtime_error("out_dir: " + dir_err);
        }
        if (!opt.stats_jsonl.empty() && !ensure_parent_dir(opt.stats_jsonl, dir_err)) {
            std::cerr << "WARN: stats parent dir: " << dir_err << "\n";
        }

        std::vector<std::unique_ptr<ReplayRuntime>> runtimes;
        runtimes.reserve(specs.size());
        for (const auto &spec : specs) {
            auto rt = std::make_unique<ReplayRuntime>();
            rt->spec = spec;
            std::cout << "[event_file_fifo_broker] opening alias=" << spec.alias
                      << " raw=" << spec.path << "\n";
            init_runtime(*rt, opt);
            std::cout << "[event_file_fifo_broker] open_result alias=" << spec.alias
                      << " opened=" << (rt->stats.opened.load() ? "true" : "false")
                      << " error=" << rt->stats.last_error << "\n";
            runtimes.push_back(std::move(rt));
        }

        std::ofstream jsonl;
        if (!opt.stats_jsonl.empty()) {
            jsonl.open(opt.stats_jsonl, std::ios::out | std::ios::trunc);
            if (!jsonl) {
                std::cerr << "WARN: failed to open stats jsonl " << opt.stats_jsonl << "\n";
            }
        }

        auto start_all = [&](bool reconfigure_for_next_loop) -> bool {
            bool any_started = false;
            for (auto &rt : runtimes) {
                if (!rt->stats.opened.load(std::memory_order_relaxed)) {
                    continue;
                }
                if (reconfigure_for_next_loop) {
                    try {
                        if (rt->stats.started.load(std::memory_order_relaxed)) {
                            rt->camera.stop();
                        }
                    } catch (...) {}
                    {
                        std::lock_guard<std::mutex> lk(rt->emitter_mu);
                        rt->emitter.finalize(false);
                        rt->emitter.advance_loop_offset();
                    }
                    rt->stats.loops_completed.fetch_add(1, std::memory_order_relaxed);
                    if (!configure_camera(*rt, opt)) {
                        rt->stats.set_error("loop configure_camera failed");
                        continue;
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(rt->emitter_mu);
                    rt->emitter.emit_hello();
                }
                const bool ok = rt->camera.start();
                rt->stats.started.store(ok, std::memory_order_relaxed);
                rt->stats.finished.store(false, std::memory_order_relaxed);
                if (!ok) {
                    rt->stats.set_error("camera.start returned false");
                } else {
                    any_started = true;
                }
            }
            return any_started;
        };

        start_all(false);

        uint64_t tick = 0;
        int64_t last_report_us = 0;
        while (!g_stop.load(std::memory_order_relaxed)) {
            bool any_running = false;
            for (auto &rt : runtimes) {
                if (rt->stats.started.load(std::memory_order_relaxed) && rt->camera.is_running()) {
                    any_running = true;
                }
            }
            const int64_t now = steady_us();
            if (last_report_us <= 0 || now - last_report_us >= static_cast<int64_t>(opt.report_ms) * 1000ll) {
                const auto json = aggregate_json(runtimes, tick++, opt);
                if (jsonl) {
                    jsonl << json << "\n";
                    jsonl.flush();
                }
                last_report_us = now;
            }
            if (!any_running) {
                if (opt.loop && !opt.inspect_only && !g_stop.load(std::memory_order_relaxed)) {
                    if (start_all(true)) {
                        continue;
                    }
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        for (auto &rt : runtimes) {
            if (rt->stats.started.load(std::memory_order_relaxed)) {
                try {
                    rt->camera.stop();
                } catch (...) {}
            }
            {
                std::lock_guard<std::mutex> lk(rt->emitter_mu);
                rt->emitter.finalize(true);
            }
            rt->writer.close();
            rt->stats.finished.store(true, std::memory_order_relaxed);
        }

        const auto final_json = aggregate_json(runtimes, tick++, opt);
        if (jsonl) {
            jsonl << final_json << "\n";
            jsonl.flush();
        }
        std::cout << final_json << "\n";

        if (!opt.allow_missing_ext_trigger) {
            bool missing = false;
            for (const auto &rt : runtimes) {
                if (rt->stats.opened.load(std::memory_order_relaxed) &&
                    rt->stats.triggers_seen.load(std::memory_order_relaxed) == 0) {
                    missing = true;
                    std::cerr << "ERROR: missing External Trigger events alias=" << rt->stats.alias
                              << " raw=" << rt->stats.raw_file << "\n";
                }
            }
            if (missing) {
                return 2;
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "event_file_fifo_broker error: " << e.what() << "\n";
        return 1;
    }
}

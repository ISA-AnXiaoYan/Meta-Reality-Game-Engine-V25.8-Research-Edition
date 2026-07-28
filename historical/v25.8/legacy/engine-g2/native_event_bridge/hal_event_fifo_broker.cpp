#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <ctime>
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
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_erc_module.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_event_trail_filter_module.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>
#include <metavision/hal/facilities/i_trigger_in.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>
#include <metavision/sdk/base/utils/error_utils.h>

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

struct Options {
    std::string serials_csv;
    std::string aliases_csv;
    std::string output_mode = "dry-run"; // dry-run or fifo
    std::string out_dir;
    std::string fifo_open_mode = "rdwr"; // rdwr, blocking, nonblock
    std::string stats_jsonl = "stats.jsonl";
    int duration_sec        = 45; // <=0 means run until SIGINT/SIGTERM.
    int slice_us            = 4000;
    int report_ms           = 1000;
    int poll_sleep_us       = 100;
    std::string wait_mode   = "poll"; // blocking or poll
    bool list_only          = false;
    bool enable_trigger_in  = false;
    bool enable_erc         = false;
    uint32_t erc_rate       = 10000000;
    bool enable_trail       = false;
    std::string trail_type  = "stc_keep_trail";
    uint32_t trail_th_us    = 10000;
    bool bypass_bias_range  = true;
    int open_retries        = 15;
    int open_retry_delay_ms = 1500;
    std::string raw_record_command_file;
    std::string raw_record_dir = "raw_records";
    std::string raw_record_prefix = "eventraw";
    int raw_record_poll_ms = 50;
    bool visual_mask_overlay_enable = false;
    std::string visual_mask_raster_template = "/dev/shm/event_human_mask_raster_{camera}.mmap";
    std::string visual_mask_overlay_name_template = "event_overlay_masked_latest_{camera}";
    std::string visual_mask_overlay_sidecar_template = "sync_ipc/event_overlay_masked_{camera}_latest.json";
    bool visual_mask_overlay_history_enable = false;
    std::string visual_mask_overlay_history_name_template = "event_overlay_masked_hist_{camera}_{slot}";
    std::string visual_mask_overlay_history_sidecar_template = "sync_ipc/event_overlay_masked_{camera}_history.json";
    int visual_mask_overlay_history_slots = 64;
    bool visual_mask_overlay_sparse_enable = false;
    std::string visual_mask_overlay_sparse_name_template = "event_overlay_sparse_hist_{camera}_{slot}";
    std::string visual_mask_overlay_sparse_sidecar_template = "sync_ipc/event_overlay_sparse_{camera}_history.json";
    int visual_mask_overlay_sparse_slots = 64;
    int visual_mask_overlay_sparse_max_points = 20000;
    std::string visual_mask_no_mask_policy = "hold_last_good";
    int visual_mask_slice_us = 33333;
    int visual_mask_accumulation_us = 33333;
    int visual_mask_publish_fps = 40;
    int visual_mask_stale_ms = 120;
    std::string visual_mask_control_json = "sync_ipc/event_mask_control.json";
    int visual_mask_control_poll_ms = 250;
    std::vector<std::pair<std::string, int>> biases;
};

std::vector<std::string> split_csv(const std::string &csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char c) {
                       return !std::isspace(c);
                   }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char c) {
                       return !std::isspace(c);
                   }).base(),
                   item.end());
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
        case '\\':
            os << "\\\\";
            break;
        case '"':
            os << "\\\"";
            break;
        case '\n':
            os << "\\n";
            break;
        case '\r':
            os << "\\r";
            break;
        case '\t':
            os << "\\t";
            break;
        default:
            os << c;
            break;
        }
    }
    return os.str();
}

int64_t steady_us() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

double process_cpu_seconds() {
    rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    const double user = static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec) / 1e6;
    const double sys  = static_cast<double>(usage.ru_stime.tv_sec) + static_cast<double>(usage.ru_stime.tv_usec) / 1e6;
    return user + sys;
}

int64_t wall_us() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

std::string connection_name(Metavision::ConnectionType t) {
    switch (t) {
    case Metavision::MIPI_LINK:
        return "MIPI";
    case Metavision::USB_LINK:
        return "USB";
    case Metavision::NETWORK_LINK:
        return "NETWORK";
    case Metavision::PROPRIETARY_LINK:
        return "PROPRIETARY";
    default:
        return "UNKNOWN";
    }
}

Metavision::I_EventTrailFilterModule::Type parse_trail_type(const std::string &s) {
    using T = Metavision::I_EventTrailFilterModule::Type;
    if (s == "trail") {
        return T::TRAIL;
    }
    if (s == "stc_cut_trail") {
        return T::STC_CUT_TRAIL;
    }
    return T::STC_KEEP_TRAIL;
}

void append_source_report(std::ostream &os) {
    try {
        os << "hal_discovery_list_begin\n";
        for (const auto &serial : Metavision::DeviceDiscovery::list()) {
            os << "serial " << serial << "\n";
        }
        os << "hal_discovery_list_end\n";
    } catch (const std::exception &e) {
        os << "hal_discovery_list_error " << e.what() << "\n";
    }

    try {
        os << "hal_available_sources_begin\n";
        for (const auto &desc : Metavision::DeviceDiscovery::list_available_sources()) {
            os << "source serial=" << desc.serial_ << " full=" << desc.get_full_serial()
               << " integrator=" << desc.integrator_name_ << " plugin=" << desc.plugin_name_
               << " connection=" << connection_name(desc.connection_) << " system_id=" << desc.system_id_ << "\n";
        }
        os << "hal_available_sources_end\n";
    } catch (const std::exception &e) {
        os << "hal_available_sources_error " << e.what() << "\n";
    }
}

std::vector<std::string> serial_candidates(const std::string &requested) {
    std::vector<std::string> out;
    if (!requested.empty()) {
        out.push_back(requested);
    }
    try {
        for (const auto &desc : Metavision::DeviceDiscovery::list_available_sources()) {
            const std::string full = desc.get_full_serial();
            if (requested.empty() || desc.serial_ == requested || full == requested) {
                if (std::find(out.begin(), out.end(), full) == out.end()) {
                    out.push_back(full);
                }
                if (std::find(out.begin(), out.end(), desc.serial_) == out.end()) {
                    out.push_back(desc.serial_);
                }
            }
        }
    } catch (...) {
    }
    if (out.empty()) {
        out.push_back("");
    }
    return out;
}

std::string clean_alias(std::string alias) {
    for (char &c : alias) {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
        if (!ok) {
            c = '_';
        }
    }
    if (alias.empty()) {
        alias = "cam";
    }
    return alias;
}

std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) {
        return b;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

std::string dirname_of(const std::string &path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

bool ensure_parent_dir(const std::string &path, std::string &err) {
    const std::string dir = dirname_of(path);
    if (dir.empty() || dir == "/") {
        return true;
    }
    std::string cur;
    if (!dir.empty() && dir[0] == '/') {
        cur = "/";
    }
    std::stringstream ss(dir);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty()) {
            continue;
        }
        if (!cur.empty() && cur.back() != '/') {
            cur += "/";
        }
        cur += part;
        struct stat st {};
        if (::stat(cur.c_str(), &st) == 0) {
            if (!S_ISDIR(st.st_mode)) {
                err = cur + " exists but is not a directory";
                return false;
            }
            continue;
        }
        if (::mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) {
            err = std::string("mkdir ") + cur + ": " + std::strerror(errno);
            return false;
        }
    }
    return true;
}

std::string replace_all(std::string s, const std::string &needle, const std::string &value) {
    size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        s.replace(pos, needle.size(), value);
        pos += value.size();
    }
    return s;
}

std::string format_camera_template(std::string tmpl, const std::string &alias) {
    tmpl = replace_all(tmpl, "{camera}", alias);
    tmpl = replace_all(tmpl, "{cam}", alias);
    tmpl = replace_all(tmpl, "{id}", alias);
    tmpl = replace_all(tmpl, "{alias}", alias);
    return tmpl;
}

std::string format_camera_slot_template(std::string tmpl, const std::string &alias, int slot) {
    tmpl = format_camera_template(std::move(tmpl), alias);
    tmpl = replace_all(tmpl, "{slot}", std::to_string(slot));
    return tmpl;
}

std::string shared_frame_path_from_name(const std::string &name_or_path) {
    if (name_or_path.empty()) {
        return "";
    }
    if (name_or_path[0] == '/') {
        return name_or_path;
    }
    if (name_or_path.size() >= 5 && name_or_path.substr(name_or_path.size() - 5) == ".mmap") {
        return join_path("/dev/shm", name_or_path);
    }
    return join_path("/dev/shm", name_or_path + ".mmap");
}

bool ensure_dir(const std::string &path, std::string &err) {
    if (path.empty()) {
        err = "empty directory";
        return false;
    }
    if (::mkdir(path.c_str(), 0775) == 0) {
        return true;
    }
    if (errno == EEXIST) {
        struct stat st {};
        if (::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            return true;
        }
        err = path + " exists but is not a directory";
        return false;
    }
    err = std::string("mkdir failed: ") + std::strerror(errno);
    return false;
}

std::string now_session_id() {
    const std::time_t now = std::time(nullptr);
    std::tm tm {};
    localtime_r(&now, &tm);
    std::ostringstream os;
    os << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return os.str();
}

int64_t file_mtime_ns(const std::string &path) {
    if (path.empty()) {
        return 0;
    }
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<int64_t>(st.st_mtimespec.tv_sec) * 1000000000ll +
           static_cast<int64_t>(st.st_mtimespec.tv_nsec);
#else
    return static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000ll +
           static_cast<int64_t>(st.st_mtim.tv_nsec);
#endif
}

bool json_number_field(const std::string &text, const std::string &key, double &value) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    const bool quoted = pos < text.size() && text[pos] == '"';
    if (quoted) {
        ++pos;
    }
    const size_t start = pos;
    while (pos < text.size()) {
        const char c = text[pos];
        if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.') {
            ++pos;
            continue;
        }
        break;
    }
    if (pos <= start) {
        return false;
    }
    try {
        value = std::stod(text.substr(start, pos - start));
        return true;
    } catch (...) {
        return false;
    }
}

bool json_int_any(const std::string &text, const std::vector<std::string> &keys, int &value) {
    for (const auto &key : keys) {
        double v = 0.0;
        if (json_number_field(text, key, v)) {
            value = static_cast<int>(v >= 0.0 ? v + 0.5 : v - 0.5);
            return true;
        }
    }
    return false;
}

bool json_int64_field(const std::string &text, const std::string &key, int64_t &value) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    const bool quoted = pos < text.size() && text[pos] == '"';
    if (quoted) {
        ++pos;
    }
    const size_t start = pos;
    if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
        ++pos;
    }
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        ++pos;
    }
    if (pos <= start || (pos == start + 1 && (text[start] == '-' || text[start] == '+'))) {
        return false;
    }
    try {
        value = std::stoll(text.substr(start, pos - start));
        return true;
    } catch (...) {
        return false;
    }
}

bool json_int64_any(const std::string &text, const std::vector<std::string> &keys, int64_t &value) {
    for (const auto &key : keys) {
        if (json_int64_field(text, key, value)) {
            return true;
        }
    }
    return false;
}

bool json_ms_any_to_us(const std::string &text, const std::vector<std::string> &keys, int &value) {
    for (const auto &key : keys) {
        double v = 0.0;
        if (json_number_field(text, key, v)) {
            const double us = v * 1000.0;
            value = static_cast<int>(us >= 0.0 ? us + 0.5 : us - 0.5);
            return true;
        }
    }
    return false;
}

std::string sanitize_filename(std::string s) {
    for (char &c : s) {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
        if (!ok) {
            c = '_';
        }
    }
    if (s.empty()) {
        s = "eventraw";
    }
    return s;
}

std::string read_text_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        return "";
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string json_string_value(const std::string &text, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return "";
    }
    const auto colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return "";
    }
    const auto first_quote = text.find('"', colon + 1);
    if (first_quote == std::string::npos) {
        return "";
    }
    std::string out;
    bool escaped = false;
    for (size_t i = first_quote + 1; i < text.size(); ++i) {
        const char c = text[i];
        if (escaped) {
            out.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return out;
        }
        out.push_back(c);
    }
    return "";
}

uint64_t json_uint_value(const std::string &text, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return 0;
    }
    const auto colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return 0;
    }
    auto pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    uint64_t value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        value = value * 10 + static_cast<uint64_t>(text[pos] - '0');
        ++pos;
    }
    return value;
}

std::vector<std::string> json_string_list_value(const std::string &text, const std::string &key) {
    std::vector<std::string> out;
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return out;
    }
    const auto colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return out;
    }
    auto pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (pos < text.size() && text[pos] == '"') {
        const std::string one = json_string_value(text, key);
        for (auto &item : split_csv(one)) {
            out.push_back(item);
        }
        return out;
    }
    if (pos >= text.size() || text[pos] != '[') {
        return out;
    }
    const auto end = text.find(']', pos + 1);
    if (end == std::string::npos) {
        return out;
    }
    size_t cur = pos + 1;
    while (cur < end) {
        const auto q = text.find('"', cur);
        if (q == std::string::npos || q >= end) {
            break;
        }
        std::string item;
        bool escaped = false;
        size_t i = q + 1;
        for (; i < end; ++i) {
            const char c = text[i];
            if (escaped) {
                item.push_back(c);
                escaped = false;
                continue;
            }
            if (c == '\\') {
                escaped = true;
                continue;
            }
            if (c == '"') {
                break;
            }
            item.push_back(c);
        }
        if (!item.empty()) {
            out.push_back(item);
        }
        cur = i + 1;
    }
    return out;
}

bool raw_record_targets_event(const std::string &text) {
    std::vector<std::string> targets = json_string_list_value(text, "targets");
    if (targets.empty()) {
        targets = json_string_list_value(text, "target");
    }
    if (targets.empty()) {
        targets = json_string_list_value(text, "record_targets");
    }
    if (targets.empty()) {
        return true;
    }
    for (auto item : targets) {
        std::transform(item.begin(), item.end(), item.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        std::replace(item.begin(), item.end(), '-', '_');
        if (item == "all" || item == "both" || item == "*" || item == "event_raw" || item == "event" ||
            item == "events" || item == "raw" || item == "raw_event" || item == "eventcam" ||
            item == "event_camera") {
            return true;
        }
    }
    return false;
}

bool prepare_fifo(const std::string &path, std::string &err) {
    struct stat st {};
    if (::lstat(path.c_str(), &st) == 0) {
        if (S_ISFIFO(st.st_mode)) {
            return true;
        }
        if (::unlink(path.c_str()) != 0) {
            err = std::string("unlink non-fifo failed: ") + std::strerror(errno);
            return false;
        }
    } else if (errno != ENOENT) {
        err = std::string("lstat failed: ") + std::strerror(errno);
        return false;
    }
    if (::mkfifo(path.c_str(), 0664) != 0 && errno != EEXIST) {
        err = std::string("mkfifo failed: ") + std::strerror(errno);
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

struct CameraStats {
    std::string alias;
    std::string requested_serial;
    std::string opened_serial;
    std::string open_attempts;
    std::string open_error;
    std::string config_log;
    std::string fifo_path;
    std::atomic<bool> opened{false};
    std::atomic<bool> started{false};
    std::atomic<bool> stream_stopped{false};
    std::atomic<bool> fifo_opened{false};
    std::atomic<uint64_t> poll_one{0};
    std::atomic<uint64_t> poll_zero{0};
    std::atomic<uint64_t> poll_neg{0};
    std::atomic<uint64_t> raw_buffers{0};
    std::atomic<uint64_t> raw_bytes{0};
    std::atomic<uint64_t> decode_calls{0};
    std::atomic<uint64_t> decode_us{0};
    std::atomic<uint64_t> callbacks{0};
    std::atomic<uint64_t> callback_empty{0};
    std::atomic<uint64_t> trigger_callbacks{0};
    std::atomic<uint64_t> events_seen{0};
    std::atomic<uint64_t> triggers_seen{0};
    std::atomic<bool> trigger_in_requested{false};
    std::atomic<bool> trigger_in_enable_attempted{false};
    std::atomic<bool> trigger_in_enable_ok{false};
    std::atomic<bool> ext_trigger_decoder_available{false};
    std::atomic<uint64_t> frames_written{0};
    std::atomic<uint64_t> events_written{0};
    std::atomic<uint64_t> triggers_written{0};
    std::atomic<uint64_t> write_errors{0};
    std::atomic<uint64_t> exceptions{0};
    std::atomic<bool> raw_record_enabled{false};
    std::atomic<bool> raw_recording{false};
    std::atomic<uint64_t> raw_record_commands_seen{0};
    std::atomic<uint64_t> raw_record_start_count{0};
    std::atomic<uint64_t> raw_record_stop_count{0};
    std::atomic<uint64_t> raw_record_error_count{0};
    std::atomic<uint64_t> raw_record_last_seq{0};

    mutable std::mutex mu;
    int64_t first_ts = -1;
    int64_t last_ts  = -1;
    int64_t first_ext_trigger_ts_us = -1;
    int64_t last_ext_trigger_ts_us = -1;
    int64_t last_ext_trigger_wall_us = -1;
    int64_t last_emit_us = -1;
    std::string last_error;
    std::string raw_record_path;
    std::string raw_record_session_id;
    std::string raw_record_last_command;
    std::string raw_record_last_error;

    void set_last_error(const std::string &msg) {
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

    void on_ext_trigger_ts(int64_t t) {
        std::lock_guard<std::mutex> lk(mu);
        if (first_ext_trigger_ts_us < 0) {
            first_ext_trigger_ts_us = t;
        }
        last_ext_trigger_ts_us = std::max<int64_t>(last_ext_trigger_ts_us, t);
        last_ext_trigger_wall_us = wall_us();
    }

    void on_emit(int64_t t) {
        std::lock_guard<std::mutex> lk(mu);
        last_emit_us = t;
    }

    void set_raw_record_state(bool recording, const std::string &path, const std::string &session_id,
                              const std::string &command) {
        raw_recording.store(recording, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mu);
        raw_record_path = path;
        raw_record_session_id = session_id;
        raw_record_last_command = command;
        if (recording) {
            raw_record_last_error.clear();
        }
    }

    void set_raw_record_error(const std::string &msg) {
        raw_record_error_count.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mu);
        raw_record_last_error = msg;
        last_error = msg;
    }

    std::string json_snapshot() const {
        std::lock_guard<std::mutex> lk(mu);
        std::ostringstream os;
        os << "{";
        os << "\"alias\":\"" << json_escape(alias) << "\",";
        os << "\"requested_serial\":\"" << json_escape(requested_serial) << "\",";
        os << "\"opened_serial\":\"" << json_escape(opened_serial) << "\",";
        os << "\"opened\":" << (opened.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"started\":" << (started.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"stream_stopped\":" << (stream_stopped.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"fifo_opened\":" << (fifo_opened.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"fifo_path\":\"" << json_escape(fifo_path) << "\",";
        os << "\"open_attempts\":\"" << json_escape(open_attempts) << "\",";
        os << "\"open_error\":\"" << json_escape(open_error) << "\",";
        os << "\"config_log\":\"" << json_escape(config_log) << "\",";
        os << "\"poll_one\":" << poll_one.load(std::memory_order_relaxed) << ",";
        os << "\"poll_zero\":" << poll_zero.load(std::memory_order_relaxed) << ",";
        os << "\"poll_neg\":" << poll_neg.load(std::memory_order_relaxed) << ",";
        os << "\"raw_buffers\":" << raw_buffers.load(std::memory_order_relaxed) << ",";
        os << "\"raw_bytes\":" << raw_bytes.load(std::memory_order_relaxed) << ",";
        os << "\"decode_calls\":" << decode_calls.load(std::memory_order_relaxed) << ",";
        os << "\"decode_us\":" << decode_us.load(std::memory_order_relaxed) << ",";
        os << "\"callbacks\":" << callbacks.load(std::memory_order_relaxed) << ",";
        os << "\"callback_empty\":" << callback_empty.load(std::memory_order_relaxed) << ",";
        os << "\"trigger_callbacks\":" << trigger_callbacks.load(std::memory_order_relaxed) << ",";
        os << "\"events_seen\":" << events_seen.load(std::memory_order_relaxed) << ",";
        os << "\"triggers_seen\":" << triggers_seen.load(std::memory_order_relaxed) << ",";
        os << "\"trigger_in_requested\":" << (trigger_in_requested.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"trigger_in_enable_attempted\":"
           << (trigger_in_enable_attempted.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"trigger_in_enable_ok\":" << (trigger_in_enable_ok.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"ext_trigger_decoder_available\":"
           << (ext_trigger_decoder_available.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"frames_written\":" << frames_written.load(std::memory_order_relaxed) << ",";
        os << "\"events_written\":" << events_written.load(std::memory_order_relaxed) << ",";
        os << "\"triggers_written\":" << triggers_written.load(std::memory_order_relaxed) << ",";
        os << "\"write_errors\":" << write_errors.load(std::memory_order_relaxed) << ",";
        os << "\"exceptions\":" << exceptions.load(std::memory_order_relaxed) << ",";
        os << "\"raw_record_enabled\":" << (raw_record_enabled.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"raw_recording\":" << (raw_recording.load(std::memory_order_relaxed) ? "true" : "false") << ",";
        os << "\"raw_record_path\":\"" << json_escape(raw_record_path) << "\",";
        os << "\"raw_record_session_id\":\"" << json_escape(raw_record_session_id) << "\",";
        os << "\"raw_record_last_command\":\"" << json_escape(raw_record_last_command) << "\",";
        os << "\"raw_record_last_error\":\"" << json_escape(raw_record_last_error) << "\",";
        os << "\"raw_record_last_seq\":" << raw_record_last_seq.load(std::memory_order_relaxed) << ",";
        os << "\"raw_record_commands_seen\":" << raw_record_commands_seen.load(std::memory_order_relaxed) << ",";
        os << "\"raw_record_start_count\":" << raw_record_start_count.load(std::memory_order_relaxed) << ",";
        os << "\"raw_record_stop_count\":" << raw_record_stop_count.load(std::memory_order_relaxed) << ",";
        os << "\"raw_record_error_count\":" << raw_record_error_count.load(std::memory_order_relaxed) << ",";
        os << "\"first_ts\":" << first_ts << ",";
        os << "\"last_ts\":" << last_ts << ",";
        os << "\"first_ext_trigger_ts_us\":" << first_ext_trigger_ts_us << ",";
        os << "\"last_ext_trigger_ts_us\":" << last_ext_trigger_ts_us << ",";
        os << "\"last_ext_trigger_wall_us\":" << last_ext_trigger_wall_us << ",";
        if (last_ext_trigger_wall_us > 0) {
            os << "\"last_ext_trigger_age_ms\":"
               << (static_cast<double>(wall_us() - last_ext_trigger_wall_us) / 1000.0) << ",";
        } else {
            os << "\"last_ext_trigger_age_ms\":null,";
        }
        os << "\"last_emit_us\":" << last_emit_us << ",";
        os << "\"last_error\":\"" << json_escape(last_error) << "\"";
        os << "}";
        return os.str();
    }
};

struct FrameWriter {
    CameraStats *stats = nullptr;
    bool dry_run = true;
    int fd = -1;

    bool open_for_camera(const Options &opt, const std::string &alias, CameraStats &st) {
        stats = &st;
        dry_run = (opt.output_mode == "dry-run");
        if (dry_run) {
            return true;
        }

        std::string err;
        if (!ensure_dir(opt.out_dir, err)) {
            st.open_error = "out_dir: " + err;
            return false;
        }
        const std::string path = join_path(opt.out_dir, "event_fifo_" + clean_alias(alias) + ".evb");
        st.fifo_path = path;
        if (!prepare_fifo(path, err)) {
            st.open_error = "prepare_fifo " + path + ": " + err;
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
            st.open_error = "open_fifo " + path + ": " + std::strerror(errno);
            return false;
        }
        st.fifo_opened.store(true, std::memory_order_relaxed);
        return true;
    }

    bool write_frame(uint32_t width, uint32_t height, uint64_t seq, uint32_t flags, int64_t slice_start_us,
                     int64_t slice_end_us, int64_t current_time_us, const std::vector<WireEventCD> &events,
                     const std::vector<WireExtTrigger> &triggers) {
        if (stats) {
            stats->frames_written.fetch_add(1, std::memory_order_relaxed);
            stats->events_written.fetch_add(static_cast<uint64_t>(events.size()), std::memory_order_relaxed);
            stats->triggers_written.fetch_add(static_cast<uint64_t>(triggers.size()), std::memory_order_relaxed);
            stats->on_emit(current_time_us);
        }
        if (dry_run) {
            return true;
        }
        if (fd < 0) {
            if (stats) {
                stats->write_errors.fetch_add(1, std::memory_order_relaxed);
                stats->set_last_error("fifo fd is not open");
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
            stats->set_last_error(err);
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
            const int64_t idx = static_cast<int64_t>(ev->t / slice_us);
            if (!advance_to(idx)) {
                return false;
            }
            current_events.push_back(WireEventCD{
                static_cast<uint16_t>(ev->x),
                static_cast<uint16_t>(ev->y),
                static_cast<int16_t>(ev->p),
                static_cast<int64_t>(ev->t),
            });
        }
        return true;
    }

    void on_triggers(const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
        for (const auto *ev = begin; ev != end; ++ev) {
            const int64_t idx = static_cast<int64_t>(ev->t / slice_us);
            pending_triggers[idx].push_back(WireExtTrigger{
                static_cast<int16_t>(ev->p),
                static_cast<int16_t>(ev->id),
                static_cast<int64_t>(ev->t),
            });
        }
    }

    void finalize() {
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
        const std::vector<WireEventCD> empty_events;
        const std::vector<WireExtTrigger> empty_triggers;
        if (writer) {
            writer->write_frame(width, height, seq++, kFlagFinal, 0, 0, 0, empty_events, empty_triggers);
        }
    }
};

constexpr uint32_t kSharedFrameMagic = 0x46524D32u;
constexpr uint32_t kSharedFrameVersion = 2u;
constexpr uint32_t kSharedFrameHeaderU32Count = 32u;
constexpr uint32_t kSharedFrameHeaderBytes = kSharedFrameHeaderU32Count * 4u;
constexpr uint32_t kSharedFrameDtypeUint8 = 1u;

#pragma pack(push, 1)
struct MaskRasterHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t seq;
    uint32_t width;
    uint32_t height;
    uint32_t data_bytes;
    uint64_t published_wall_us;
};
#pragma pack(pop)

static_assert(sizeof(MaskRasterHeader) == 44, "MaskRasterHeader layout must match Python publisher");

struct SharedFrameMmapWriter {
    std::string path;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 1;
    uint32_t stride = 0;
    uint32_t data_bytes = 0;
    size_t total_bytes = 0;
    int fd = -1;
    uint8_t *base = nullptr;
    uint32_t *header = nullptr;
    uint64_t frame_id = 0;

    bool open(const std::string &name_or_path, uint32_t w, uint32_t h, uint32_t ch, std::string &err) {
        width = w;
        height = h;
        channels = std::max<uint32_t>(1, ch);
        stride = width * channels;
        data_bytes = height * stride;
        total_bytes = static_cast<size_t>(kSharedFrameHeaderBytes) + 2u * static_cast<size_t>(data_bytes);
        path = shared_frame_path_from_name(name_or_path);
        if (path.empty()) {
            err = "empty shared frame path";
            return false;
        }
        if (!ensure_parent_dir(path, err)) {
            return false;
        }
        ::unlink(path.c_str());
        fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            err = "open " + path + ": " + std::strerror(errno);
            return false;
        }
        if (::ftruncate(fd, static_cast<off_t>(total_bytes)) != 0) {
            err = "ftruncate " + path + ": " + std::strerror(errno);
            close();
            return false;
        }
        void *mm = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mm == MAP_FAILED) {
            err = "mmap " + path + ": " + std::strerror(errno);
            base = nullptr;
            close();
            return false;
        }
        base = static_cast<uint8_t *>(mm);
        // The mmap stays valid after closing the file descriptor; keeping one fd
        // per history slot can exceed the default process fd limit.
        ::close(fd);
        fd = -1;
        header = reinterpret_cast<uint32_t *>(base);
        std::fill(header, header + kSharedFrameHeaderU32Count, 0u);
        header[0] = kSharedFrameMagic;
        header[1] = kSharedFrameVersion;
        header[2] = width;
        header[3] = height;
        header[4] = channels;
        header[5] = kSharedFrameDtypeUint8;
        header[6] = stride;
        header[7] = 0;
        header[8] = 0;
        header[9] = data_bytes;
        return true;
    }

    bool is_open() const {
        return base != nullptr && header != nullptr;
    }

    uint8_t *slot_ptr(uint32_t slot) {
        return base + kSharedFrameHeaderBytes + static_cast<size_t>(slot % 2u) * static_cast<size_t>(data_bytes);
    }

    bool write(const std::vector<uint8_t> &frame, int64_t timestamp_us) {
        if (!is_open() || frame.size() != static_cast<size_t>(data_bytes)) {
            return false;
        }
        const uint32_t active = header[8] <= 1u ? header[8] : 0u;
        const uint32_t write_slot = 1u - active;
        const uint32_t ready_idx = write_slot == 0 ? 10u : 14u;
        const uint32_t fid_idx = write_slot == 0 ? 11u : 15u;
        const uint32_t ts_lo_idx = write_slot == 0 ? 12u : 16u;
        const uint32_t ts_hi_idx = write_slot == 0 ? 13u : 17u;
        header[7] = write_slot;
        header[ready_idx] = 0u;
        std::memcpy(slot_ptr(write_slot), frame.data(), frame.size());
        ++frame_id;
        const uint64_t ts = static_cast<uint64_t>(std::max<int64_t>(0, timestamp_us));
        header[fid_idx] = static_cast<uint32_t>(frame_id & 0xFFFFFFFFu);
        header[ts_lo_idx] = static_cast<uint32_t>(ts & 0xFFFFFFFFu);
        header[ts_hi_idx] = static_cast<uint32_t>((ts >> 32) & 0xFFFFFFFFu);
        header[ready_idx] = 1u;
        header[8] = write_slot;
        return true;
    }

    void close() {
        if (base) {
            ::munmap(base, total_bytes);
            base = nullptr;
            header = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
};

constexpr uint64_t kSparseOverlayMagic = 0x455653503031ull; // EVSP01
constexpr uint64_t kSparseOverlayVersion = 1ull;
constexpr uint64_t kSparseOverlayHeaderU64Count = 64ull;
constexpr uint64_t kSparseOverlayHeaderBytes = kSparseOverlayHeaderU64Count * 8ull;

#pragma pack(push, 1)
struct SparseOverlayPoint {
    uint16_t x;
    uint16_t y;
    uint16_t count;
    uint8_t intensity;
    int32_t dt_center_us;
    uint8_t polarity;
    uint8_t reserved[4];
};
#pragma pack(pop)

static_assert(sizeof(SparseOverlayPoint) == 16, "SparseOverlayPoint must stay 16 bytes");

struct SparsePointsMmapWriter {
    std::string path;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t max_points = 0;
    size_t data_bytes = 0;
    size_t total_bytes = 0;
    int fd = -1;
    uint8_t *base = nullptr;
    uint64_t *header = nullptr;
    SparseOverlayPoint *points = nullptr;
    uint64_t frame_id = 0;

    bool open(const std::string &name_or_path, uint32_t w, uint32_t h, uint32_t max_pts, std::string &err) {
        width = w;
        height = h;
        max_points = std::max<uint32_t>(1, max_pts);
        data_bytes = static_cast<size_t>(max_points) * sizeof(SparseOverlayPoint);
        total_bytes = static_cast<size_t>(kSparseOverlayHeaderBytes) + data_bytes;
        path = shared_frame_path_from_name(name_or_path);
        if (path.empty()) {
            err = "empty sparse overlay path";
            return false;
        }
        if (!ensure_parent_dir(path, err)) {
            return false;
        }
        ::unlink(path.c_str());
        fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            err = "open " + path + ": " + std::strerror(errno);
            return false;
        }
        if (::ftruncate(fd, static_cast<off_t>(total_bytes)) != 0) {
            err = "ftruncate " + path + ": " + std::strerror(errno);
            close();
            return false;
        }
        void *mm = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mm == MAP_FAILED) {
            err = "mmap " + path + ": " + std::strerror(errno);
            base = nullptr;
            close();
            return false;
        }
        base = static_cast<uint8_t *>(mm);
        ::close(fd);
        fd = -1;
        header = reinterpret_cast<uint64_t *>(base);
        points = reinterpret_cast<SparseOverlayPoint *>(base + kSparseOverlayHeaderBytes);
        std::fill(header, header + kSparseOverlayHeaderU64Count, 0ull);
        header[0] = kSparseOverlayMagic;
        header[1] = kSparseOverlayVersion;
        header[2] = kSparseOverlayHeaderBytes;
        header[3] = sizeof(SparseOverlayPoint);
        header[4] = max_points;
        header[6] = width;
        header[7] = height;
        return true;
    }

    bool is_open() const {
        return base != nullptr && header != nullptr && points != nullptr;
    }

    bool write(
        const std::vector<SparseOverlayPoint> &src,
        int64_t start_us,
        int64_t end_us,
        int64_t center_us,
        int64_t published_wall_us,
        uint64_t dropped_points
    ) {
        if (!is_open()) {
            return false;
        }
        const uint64_t n = std::min<uint64_t>(static_cast<uint64_t>(src.size()), static_cast<uint64_t>(max_points));
        const uint64_t seq = header[15];
        header[15] = (seq % 2ull == 0ull) ? (seq + 1ull) : (seq + 2ull);
        if (n > 0) {
            std::memcpy(points, src.data(), static_cast<size_t>(n) * sizeof(SparseOverlayPoint));
        }
        ++frame_id;
        header[5] = n;
        header[8] = frame_id;
        header[9] = static_cast<uint64_t>(std::max<int64_t>(0, start_us));
        header[10] = static_cast<uint64_t>(std::max<int64_t>(0, end_us));
        header[11] = static_cast<uint64_t>(std::max<int64_t>(0, center_us));
        header[12] = static_cast<uint64_t>(std::max<int64_t>(0, published_wall_us));
        header[13] = dropped_points;
        header[15] = header[15] + 1ull;
        return true;
    }

    void close() {
        if (base) {
            ::munmap(base, total_bytes);
            base = nullptr;
            header = nullptr;
            points = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
};

struct MaskRasterReader {
    std::string path;
    int fd = -1;
    uint8_t *base = nullptr;
    size_t file_size = 0;
    uint64_t inode = 0;
    uint64_t last_seq = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t data_bytes = 0;
    uint64_t published_wall_us = 0;
    std::vector<uint8_t> mask;
    std::string last_error = "not_open";
    uint64_t open_count = 0;
    uint64_t read_count = 0;
    uint64_t skip_count = 0;

    void close() {
        if (base) {
            ::munmap(base, file_size);
            base = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        file_size = 0;
        inode = 0;
    }

    bool open_if_needed(const std::string &p) {
        if (base) {
            struct stat st {};
            if (::stat(p.c_str(), &st) == 0) {
                const uint64_t current_inode = static_cast<uint64_t>(st.st_ino);
                const size_t current_size = static_cast<size_t>(std::max<off_t>(0, st.st_size));
                if ((inode != 0 && current_inode != inode) || (current_size != 0 && current_size != file_size)) {
                    close();
                } else {
                    return true;
                }
            } else {
                return true;
            }
        }
        if (base) {
            return true;
        }
        path = p;
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            last_error = "open: " + std::string(std::strerror(errno));
            ++skip_count;
            return false;
        }
        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(MaskRasterHeader))) {
            last_error = "bad_size";
            close();
            ++skip_count;
            return false;
        }
        file_size = static_cast<size_t>(st.st_size);
        inode = static_cast<uint64_t>(st.st_ino);
        void *mm = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (mm == MAP_FAILED) {
            last_error = "mmap: " + std::string(std::strerror(errno));
            base = nullptr;
            close();
            ++skip_count;
            return false;
        }
        base = static_cast<uint8_t *>(mm);
        ++open_count;
        return true;
    }

    bool refresh(const std::string &p, bool require_new) {
        if (!open_if_needed(p) || !base) {
            return false;
        }
        MaskRasterHeader h1 {};
        std::memcpy(&h1, base, sizeof(MaskRasterHeader));
        if (std::memcmp(h1.magic, "HMRSTR1\0", 8) != 0 || h1.version != 1u ||
            h1.header_size != sizeof(MaskRasterHeader)) {
            last_error = "bad_header";
            ++skip_count;
            return false;
        }
        if (h1.seq == 0 || (h1.seq & 1u)) {
            last_error = "writer_busy";
            ++skip_count;
            return false;
        }
        if (require_new && h1.seq == last_seq) {
            ++skip_count;
            return false;
        }
        const uint32_t n = h1.data_bytes;
        if (n == 0 || sizeof(MaskRasterHeader) + static_cast<size_t>(n) > file_size) {
            last_error = "bad_payload_len";
            ++skip_count;
            return false;
        }
        std::vector<uint8_t> tmp(n);
        std::memcpy(tmp.data(), base + sizeof(MaskRasterHeader), n);
        MaskRasterHeader h2 {};
        std::memcpy(&h2, base, sizeof(MaskRasterHeader));
        if (h1.seq != h2.seq || (h2.seq & 1u) || h1.data_bytes != h2.data_bytes) {
            last_error = "unstable_read";
            ++skip_count;
            return false;
        }
        width = h2.width;
        height = h2.height;
        data_bytes = h2.data_bytes;
        published_wall_us = h2.published_wall_us;
        last_seq = h2.seq;
        mask.swap(tmp);
        last_error.clear();
        ++read_count;
        return true;
    }

    bool has_mask_for(uint32_t w, uint32_t h) const {
        if (mask.empty() || width != w || height != h || data_bytes != w * h) {
            return false;
        }
        return true;
    }

    bool valid_for(uint32_t w, uint32_t h, int stale_ms) const {
        if (!has_mask_for(w, h)) {
            return false;
        }
        if (published_wall_us == 0) {
            return true;
        }
        const int64_t age_us = wall_us() - static_cast<int64_t>(published_wall_us);
        return age_us <= static_cast<int64_t>(std::max(1, stale_ms)) * 1000;
    }
};

struct VisualMaskOverlay {
    bool enabled = false;
    std::string alias;
    uint32_t width = 0;
    uint32_t height = 0;
    int slice_us = 33333;
    int accumulation_us = 33333;
    int publish_interval_us = 25000;
    int stale_ms = 120;
    std::string no_mask_policy = "hold_last_good";
    std::string control_path;
    int control_poll_ms = 250;
    int64_t control_last_poll_steady_us = 0;
    int64_t control_mtime_ns = 0;
    int64_t control_seq = 0;
    uint64_t control_apply_count = 0;
    std::string control_source = "launch_profile";
    std::string control_error;
    std::string mask_path;
    std::string sidecar_path;
    SharedFrameMmapWriter writer;
    bool history_enable = false;
    std::string history_sidecar_path;
    int history_slots = 0;
    int history_cursor = 0;
    struct HistoryEntry {
        int slot = -1;
        std::string shm_name;
        std::string path;
        uint64_t frame_id = 0;
        int64_t event_ts_us = 0;
        int64_t event_window_start_us = 0;
        int64_t event_window_end_us = 0;
        int64_t event_window_center_us = 0;
        int64_t published_wall_us = 0;
        uint64_t nonzero_pixels = 0;
        bool valid = false;
    };
    std::vector<SharedFrameMmapWriter> history_writers;
    std::vector<HistoryEntry> history_entries;
    uint64_t history_write_count = 0;
    uint64_t history_overwrite_count = 0;
    bool sparse_enable = false;
    std::string sparse_sidecar_path;
    int sparse_slots = 0;
    int sparse_cursor = 0;
    int sparse_max_points = 0;
    struct SparseHistoryEntry {
        int slot = -1;
        std::string shm_name;
        std::string path;
        uint64_t frame_id = 0;
        int64_t event_window_start_us = 0;
        int64_t event_window_end_us = 0;
        int64_t event_window_center_us = 0;
        int64_t published_wall_us = 0;
        uint64_t point_count = 0;
        uint64_t dropped_point_count = 0;
        bool valid = false;
    };
    std::vector<SparsePointsMmapWriter> sparse_writers;
    std::vector<SparseHistoryEntry> sparse_entries;
    std::vector<SparseOverlayPoint> sparse_points_tmp;
    uint64_t sparse_write_count = 0;
    uint64_t sparse_overwrite_count = 0;
    uint64_t sparse_last_point_count = 0;
    uint64_t sparse_last_dropped_points = 0;
    uint64_t sparse_total_dropped_points = 0;
    MaskRasterReader mask_reader;
    std::deque<WireEventCD> window_events;
    std::vector<uint8_t> frame;
    int64_t current_bucket = -1;
    int64_t last_publish_ts = -1;
    uint64_t raw_events = 0;
    uint64_t filtered_events = 0;
    uint64_t dropped_events = 0;
    uint64_t publish_count = 0;
    uint64_t no_mask_event_count = 0;
    uint64_t mask_refresh_count = 0;
    uint64_t bucket_count = 0;
    uint64_t last_raw_count = 0;
    uint64_t last_filtered_count = 0;
    uint64_t last_drop_count = 0;
    uint64_t last_nonzero_pixels = 0;
    std::string last_error;
    std::string last_filter_policy = "disabled";

    bool init(const Options &opt, const std::string &cam_alias, uint32_t w, uint32_t h, std::string &err) {
        enabled = bool(opt.visual_mask_overlay_enable);
        alias = cam_alias;
        width = w;
        height = h;
        if (!enabled) {
            last_filter_policy = "disabled";
            return true;
        }
        slice_us = std::max(1000, opt.visual_mask_slice_us);
        accumulation_us = std::max(slice_us, opt.visual_mask_accumulation_us);
        publish_interval_us = 1000000 / std::max(1, opt.visual_mask_publish_fps);
        stale_ms = std::max(1, opt.visual_mask_stale_ms);
        no_mask_policy = opt.visual_mask_no_mask_policy.empty() ? "hold_last_good" : opt.visual_mask_no_mask_policy;
        control_path = opt.visual_mask_control_json;
        control_poll_ms = std::max(50, opt.visual_mask_control_poll_ms);
        mask_path = format_camera_template(opt.visual_mask_raster_template, alias);
        sidecar_path = format_camera_template(opt.visual_mask_overlay_sidecar_template, alias);
        const std::string overlay_name = format_camera_template(opt.visual_mask_overlay_name_template, alias);
        if (!writer.open(overlay_name, width, height, 1, err)) {
            last_error = err;
            return false;
        }
        history_enable = bool(opt.visual_mask_overlay_history_enable);
        history_slots = history_enable ? std::max(1, opt.visual_mask_overlay_history_slots) : 0;
        history_sidecar_path = history_enable ? format_camera_template(opt.visual_mask_overlay_history_sidecar_template, alias) : "";
        if (history_enable) {
            history_writers.resize(static_cast<size_t>(history_slots));
            history_entries.resize(static_cast<size_t>(history_slots));
            for (int i = 0; i < history_slots; ++i) {
                const std::string hist_name = format_camera_slot_template(opt.visual_mask_overlay_history_name_template, alias, i);
                if (!history_writers[static_cast<size_t>(i)].open(hist_name, width, height, 1, err)) {
                    last_error = err;
                    return false;
                }
                history_entries[static_cast<size_t>(i)].slot = i;
                history_entries[static_cast<size_t>(i)].shm_name = hist_name;
                history_entries[static_cast<size_t>(i)].path = history_writers[static_cast<size_t>(i)].path;
            }
        }
        sparse_enable = bool(opt.visual_mask_overlay_sparse_enable);
        sparse_slots = sparse_enable ? std::max(1, opt.visual_mask_overlay_sparse_slots) : 0;
        sparse_max_points = sparse_enable ? std::max(1, opt.visual_mask_overlay_sparse_max_points) : 0;
        sparse_sidecar_path = sparse_enable ? format_camera_template(opt.visual_mask_overlay_sparse_sidecar_template, alias) : "";
        if (sparse_enable) {
            sparse_writers.resize(static_cast<size_t>(sparse_slots));
            sparse_entries.resize(static_cast<size_t>(sparse_slots));
            sparse_points_tmp.reserve(static_cast<size_t>(std::min(65536, std::max(1, sparse_max_points))));
            for (int i = 0; i < sparse_slots; ++i) {
                const std::string sparse_name = format_camera_slot_template(opt.visual_mask_overlay_sparse_name_template, alias, i);
                if (!sparse_writers[static_cast<size_t>(i)].open(
                        sparse_name,
                        width,
                        height,
                        static_cast<uint32_t>(sparse_max_points),
                        err)) {
                    last_error = err;
                    return false;
                }
                sparse_entries[static_cast<size_t>(i)].slot = i;
                sparse_entries[static_cast<size_t>(i)].shm_name = sparse_name;
                sparse_entries[static_cast<size_t>(i)].path = sparse_writers[static_cast<size_t>(i)].path;
            }
        }
        frame.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0u);
        last_filter_policy = "waiting_mask";
        return true;
    }

    void apply_timing_control(int new_slice_us, int new_accumulation_us, int64_t seq, const std::string &source) {
        new_slice_us = std::max(1000, new_slice_us);
        new_accumulation_us = std::max(new_slice_us, new_accumulation_us);
        const bool changed = new_slice_us != slice_us || new_accumulation_us != accumulation_us;
        slice_us = new_slice_us;
        accumulation_us = new_accumulation_us;
        control_seq = seq;
        control_source = source.empty() ? "event_mask_control_json" : source;
        control_error.clear();
        if (changed) {
            ++control_apply_count;
            current_bucket = -1;
            last_filter_policy = "timing_hot_updated";
        }
    }

    void poll_control() {
        if (!enabled || control_path.empty()) {
            return;
        }
        const int64_t now_us = steady_us();
        if (control_last_poll_steady_us > 0 &&
            now_us - control_last_poll_steady_us < static_cast<int64_t>(std::max(50, control_poll_ms)) * 1000ll) {
            return;
        }
        control_last_poll_steady_us = now_us;
        const int64_t mtime_ns = file_mtime_ns(control_path);
        if (mtime_ns <= 0) {
            control_error = "control_file_missing";
            return;
        }
        if (mtime_ns == control_mtime_ns) {
            return;
        }
        control_mtime_ns = mtime_ns;
        std::ifstream in(control_path);
        if (!in) {
            control_error = "open_control_failed";
            return;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();
        int64_t seq_i = control_seq;
        json_int64_any(text, {"seq", "control_seq"}, seq_i);
        int new_slice = slice_us;
        int new_accumulation = accumulation_us;
        const bool has_slice =
            json_int_any(text, {"event_visual_mask_slice_us", "visual_mask_slice_us"}, new_slice) ||
            json_ms_any_to_us(text, {"event_visual_mask_slice_ms", "visual_mask_slice_ms"}, new_slice);
        const bool has_accumulation =
            json_int_any(text, {"event_visual_mask_accumulation_us", "visual_mask_accumulation_us", "visual_accumulation_us"}, new_accumulation) ||
            json_ms_any_to_us(text, {"event_visual_mask_accumulation_ms", "visual_mask_accumulation_ms", "visual_accumulation_ms"}, new_accumulation);
        if (!has_slice && !has_accumulation) {
            control_error.clear();
            return;
        }
        apply_timing_control(new_slice, new_accumulation, seq_i, "event_mask_control_json");
    }

    void refresh_mask(bool require_new) {
        if (!enabled) {
            return;
        }
        if (mask_reader.refresh(mask_path, require_new)) {
            ++mask_refresh_count;
        }
    }

    bool mask_valid() const {
        return mask_reader.valid_for(width, height, stale_ms);
    }

    bool mask_data_available() const {
        return mask_reader.has_mask_for(width, height);
    }

    bool mask_hold_active() const {
        return no_mask_policy == "hold_last_good" && mask_data_available() && !mask_valid();
    }

    bool keep_without_mask() const {
        return no_mask_policy == "raw_debug" || no_mask_policy == "raw";
    }

    void on_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (!enabled) {
            return;
        }
        uint64_t raw_local = 0;
        uint64_t kept_local = 0;
        uint64_t drop_local = 0;
        for (const auto *ev = begin; ev != end; ++ev) {
            const int64_t bucket = static_cast<int64_t>(ev->t / slice_us);
            if (bucket != current_bucket) {
                current_bucket = bucket;
                ++bucket_count;
                refresh_mask(true);
            }
            ++raw_events;
            ++raw_local;
            bool keep = true;
            const bool valid_mask = mask_valid();
            const bool hold_mask = !valid_mask && no_mask_policy == "hold_last_good" && mask_data_available();
            if (valid_mask || hold_mask) {
                const uint32_t x = static_cast<uint32_t>(ev->x);
                const uint32_t y = static_cast<uint32_t>(ev->y);
                if (x < width && y < height) {
                    const size_t idx = static_cast<size_t>(y) * width + x;
                    keep = !(idx < mask_reader.mask.size() && mask_reader.mask[idx] > 0);
                }
                last_filter_policy = valid_mask ? "mask_raster_visual_mask" : "hold_last_good_mask";
            } else {
                keep = keep_without_mask();
                ++no_mask_event_count;
                last_filter_policy = keep ? "raw_debug_no_mask" : "empty_no_mask";
            }
            if (keep) {
                window_events.push_back(WireEventCD{
                    static_cast<uint16_t>(ev->x),
                    static_cast<uint16_t>(ev->y),
                    static_cast<int16_t>(ev->p),
                    static_cast<int64_t>(ev->t),
                });
                ++filtered_events;
                ++kept_local;
            } else {
                ++dropped_events;
                ++drop_local;
            }
        }
        last_raw_count = raw_local;
        last_filtered_count = kept_local;
        last_drop_count = drop_local;
        if (end != begin) {
            publish_if_due((end - 1)->t);
        }
    }

    void trim_window(int64_t latest_ts) {
        const int64_t cutoff = latest_ts - static_cast<int64_t>(accumulation_us);
        while (!window_events.empty() && window_events.front().t < cutoff) {
            window_events.pop_front();
        }
    }

    void publish_if_due(int64_t latest_ts) {
        if (!enabled || !writer.is_open()) {
            return;
        }
        trim_window(latest_ts);
        if (last_publish_ts >= 0 && latest_ts - last_publish_ts < publish_interval_us) {
            return;
        }
        last_publish_ts = latest_ts;
        std::fill(frame.begin(), frame.end(), 0u);
        uint64_t nonzero = 0;
        for (const auto &ev : window_events) {
            if (ev.x >= width || ev.y >= height) {
                continue;
            }
            const size_t idx = static_cast<size_t>(ev.y) * width + ev.x;
            if (idx >= frame.size()) {
                continue;
            }
            if (frame[idx] == 0u) {
                ++nonzero;
            }
            frame[idx] = 255u;
        }
        last_nonzero_pixels = nonzero;
        if (!writer.write(frame, latest_ts)) {
            last_error = "shared_frame_write_failed";
            return;
        }
        write_history_frame(latest_ts, nonzero);
        write_sparse_history_frame(latest_ts);
        ++publish_count;
        write_sidecar(latest_ts);
    }

    void write_history_frame(int64_t latest_ts, uint64_t nonzero) {
        if (!history_enable || history_writers.empty()) {
            return;
        }
        const int slot = history_cursor % std::max(1, history_slots);
        auto &hw = history_writers[static_cast<size_t>(slot)];
        auto &entry = history_entries[static_cast<size_t>(slot)];
        if (entry.valid) {
            ++history_overwrite_count;
        }
        if (!hw.write(frame, latest_ts)) {
            last_error = "history_shared_frame_write_failed";
            return;
        }
        const int64_t end_us = latest_ts;
        const int64_t start_us = latest_ts - static_cast<int64_t>(accumulation_us);
        const int64_t center_us = start_us + static_cast<int64_t>(accumulation_us / 2);
        entry.slot = slot;
        entry.path = hw.path;
        entry.frame_id = hw.frame_id;
        entry.event_ts_us = latest_ts;
        entry.event_window_start_us = start_us;
        entry.event_window_end_us = end_us;
        entry.event_window_center_us = center_us;
        entry.published_wall_us = wall_us();
        entry.nonzero_pixels = nonzero;
        entry.valid = true;
        ++history_write_count;
        history_cursor = (slot + 1) % std::max(1, history_slots);
    }

    void write_sparse_history_frame(int64_t latest_ts) {
        if (!sparse_enable || sparse_writers.empty()) {
            sparse_last_point_count = 0;
            sparse_last_dropped_points = 0;
            return;
        }
        struct SparseAgg {
            uint32_t count = 0;
            int64_t latest_t = 0;
            int polarity_sum = 0;
        };
        const int64_t end_us = latest_ts;
        const int64_t start_us = latest_ts - static_cast<int64_t>(accumulation_us);
        const int64_t center_us = start_us + static_cast<int64_t>(accumulation_us / 2);
        std::unordered_map<uint32_t, SparseAgg> agg;
        const size_t reserve_n = std::min<size_t>(
            static_cast<size_t>(std::max(1, sparse_max_points)) * 2u,
            std::max<size_t>(16u, window_events.size())
        );
        agg.reserve(reserve_n);
        uint64_t dropped_unique = 0;
        for (auto it = window_events.rbegin(); it != window_events.rend(); ++it) {
            const auto &ev = *it;
            if (ev.x >= width || ev.y >= height) {
                continue;
            }
            const uint32_t key = static_cast<uint32_t>(ev.y) * width + static_cast<uint32_t>(ev.x);
            auto found = agg.find(key);
            if (found == agg.end()) {
                if (agg.size() >= static_cast<size_t>(std::max(1, sparse_max_points))) {
                    ++dropped_unique;
                    continue;
                }
                SparseAgg item;
                item.count = 1u;
                item.latest_t = ev.t;
                item.polarity_sum = ev.p >= 0 ? 1 : -1;
                agg.emplace(key, item);
            } else {
                auto &item = found->second;
                item.count = std::min<uint32_t>(65535u, item.count + 1u);
                if (ev.t > item.latest_t) {
                    item.latest_t = ev.t;
                }
                item.polarity_sum += ev.p >= 0 ? 1 : -1;
            }
        }
        sparse_points_tmp.clear();
        sparse_points_tmp.reserve(std::min<size_t>(agg.size(), static_cast<size_t>(std::max(1, sparse_max_points))));
        for (const auto &kv : agg) {
            const uint32_t key = kv.first;
            const SparseAgg &item = kv.second;
            SparseOverlayPoint pt {};
            pt.x = static_cast<uint16_t>(key % width);
            pt.y = static_cast<uint16_t>(key / width);
            pt.count = static_cast<uint16_t>(std::min<uint32_t>(65535u, std::max<uint32_t>(1u, item.count)));
            pt.intensity = static_cast<uint8_t>(std::min<uint32_t>(255u, 32u + item.count * 16u));
            const int64_t dt = item.latest_t - center_us;
            pt.dt_center_us = static_cast<int32_t>(std::max<int64_t>(-2147483647ll, std::min<int64_t>(2147483647ll, dt)));
            pt.polarity = static_cast<uint8_t>(item.polarity_sum >= 0 ? 1u : 0u);
            sparse_points_tmp.push_back(pt);
        }

        const int slot = sparse_cursor % std::max(1, sparse_slots);
        auto &sw = sparse_writers[static_cast<size_t>(slot)];
        auto &entry = sparse_entries[static_cast<size_t>(slot)];
        if (entry.valid) {
            ++sparse_overwrite_count;
        }
        const int64_t published = wall_us();
        if (!sw.write(sparse_points_tmp, start_us, end_us, center_us, published, dropped_unique)) {
            last_error = "sparse_shared_frame_write_failed";
            return;
        }
        entry.slot = slot;
        entry.path = sw.path;
        entry.frame_id = sw.frame_id;
        entry.event_window_start_us = start_us;
        entry.event_window_end_us = end_us;
        entry.event_window_center_us = center_us;
        entry.published_wall_us = published;
        entry.point_count = static_cast<uint64_t>(sparse_points_tmp.size());
        entry.dropped_point_count = dropped_unique;
        entry.valid = true;
        sparse_last_point_count = entry.point_count;
        sparse_last_dropped_points = dropped_unique;
        sparse_total_dropped_points += dropped_unique;
        ++sparse_write_count;
        sparse_cursor = (slot + 1) % std::max(1, sparse_slots);
    }

    void write_history_sidecar() {
        if (!history_enable || history_sidecar_path.empty()) {
            return;
        }
        std::string err;
        if (!ensure_parent_dir(history_sidecar_path, err)) {
            last_error = err;
            return;
        }
        const std::string tmp = history_sidecar_path + ".tmp";
        std::ofstream out(tmp);
        if (!out) {
            last_error = "open_history_sidecar_failed";
            return;
        }
        out << "{";
        out << "\"schema_version\":1,";
        out << "\"source\":\"hal_visual_mask_overlay_history\",";
        out << "\"camera\":\"" << json_escape(alias) << "\",";
        out << "\"published_wall_us\":" << wall_us() << ",";
        out << "\"history_enable\":" << (history_enable ? "true" : "false") << ",";
        out << "\"history_slots\":" << history_slots << ",";
        out << "\"history_cursor\":" << history_cursor << ",";
        out << "\"history_write_count\":" << history_write_count << ",";
        out << "\"history_overwrite_count\":" << history_overwrite_count << ",";
        out << "\"visual_mask_slice_us\":" << slice_us << ",";
        out << "\"visual_accumulation_us\":" << accumulation_us << ",";
        int valid_count = 0;
        int64_t oldest_center_us = 0;
        int64_t newest_center_us = 0;
        for (const auto &entry : history_entries) {
            if (!entry.valid) {
                continue;
            }
            ++valid_count;
            if (oldest_center_us == 0 || entry.event_window_center_us < oldest_center_us) {
                oldest_center_us = entry.event_window_center_us;
            }
            if (newest_center_us == 0 || entry.event_window_center_us > newest_center_us) {
                newest_center_us = entry.event_window_center_us;
            }
        }
        const int64_t history_coverage_us = (oldest_center_us > 0 && newest_center_us >= oldest_center_us)
            ? (newest_center_us - oldest_center_us)
            : 0;
        out << "\"history_valid_count\":" << valid_count << ",";
        out << "\"history_oldest_center_us\":" << oldest_center_us << ",";
        out << "\"history_newest_center_us\":" << newest_center_us << ",";
        out << "\"history_coverage_us\":" << history_coverage_us << ",";
        out << "\"entries\":[";
        bool first = true;
        for (const auto &entry : history_entries) {
            if (!entry.valid) {
                continue;
            }
            if (!first) {
                out << ",";
            }
            first = false;
            out << "{";
            out << "\"slot\":" << entry.slot << ",";
            out << "\"shm_name\":\"" << json_escape(entry.shm_name) << "\",";
            out << "\"path\":\"" << json_escape(entry.path) << "\",";
            out << "\"frame_id\":" << entry.frame_id << ",";
            out << "\"event_ts_us\":" << entry.event_ts_us << ",";
            out << "\"event_window_start_us\":" << entry.event_window_start_us << ",";
            out << "\"event_window_end_us\":" << entry.event_window_end_us << ",";
            out << "\"event_window_center_us\":" << entry.event_window_center_us << ",";
            out << "\"published_wall_us\":" << entry.published_wall_us << ",";
            out << "\"overlay_nonzero_pixels\":" << entry.nonzero_pixels << ",";
            out << "\"overlay_max_luma\":" << (entry.nonzero_pixels > 0 ? 255 : 0) << ",";
            out << "\"valid\":" << (entry.valid ? "true" : "false");
            out << "}";
        }
        out << "],";
        out << "\"last_error\":\"" << json_escape(last_error) << "\"";
        out << "}\n";
        out.close();
        ::rename(tmp.c_str(), history_sidecar_path.c_str());
    }

    void write_sparse_history_sidecar() {
        if (!sparse_enable || sparse_sidecar_path.empty()) {
            return;
        }
        std::string err;
        if (!ensure_parent_dir(sparse_sidecar_path, err)) {
            last_error = err;
            return;
        }
        const std::string tmp = sparse_sidecar_path + ".tmp";
        std::ofstream out(tmp);
        if (!out) {
            last_error = "open_sparse_sidecar_failed";
            return;
        }
        int valid_count = 0;
        int64_t oldest_center_us = 0;
        int64_t newest_center_us = 0;
        uint64_t total_points = 0;
        for (const auto &entry : sparse_entries) {
            if (!entry.valid) {
                continue;
            }
            ++valid_count;
            total_points += entry.point_count;
            if (oldest_center_us == 0 || entry.event_window_center_us < oldest_center_us) {
                oldest_center_us = entry.event_window_center_us;
            }
            if (newest_center_us == 0 || entry.event_window_center_us > newest_center_us) {
                newest_center_us = entry.event_window_center_us;
            }
        }
        const int64_t coverage_us = (oldest_center_us > 0 && newest_center_us >= oldest_center_us)
            ? (newest_center_us - oldest_center_us)
            : 0;
        out << "{";
        out << "\"schema_version\":1,";
        out << "\"source\":\"hal_visual_mask_overlay_sparse_history\",";
        out << "\"camera\":\"" << json_escape(alias) << "\",";
        out << "\"published_wall_us\":" << wall_us() << ",";
        out << "\"sparse_enable\":" << (sparse_enable ? "true" : "false") << ",";
        out << "\"sparse_slots\":" << sparse_slots << ",";
        out << "\"sparse_cursor\":" << sparse_cursor << ",";
        out << "\"sparse_max_points\":" << sparse_max_points << ",";
        out << "\"sparse_write_count\":" << sparse_write_count << ",";
        out << "\"sparse_overwrite_count\":" << sparse_overwrite_count << ",";
        out << "\"sparse_last_point_count\":" << sparse_last_point_count << ",";
        out << "\"sparse_last_dropped_points\":" << sparse_last_dropped_points << ",";
        out << "\"sparse_total_dropped_points\":" << sparse_total_dropped_points << ",";
        out << "\"sparse_valid_count\":" << valid_count << ",";
        out << "\"sparse_oldest_center_us\":" << oldest_center_us << ",";
        out << "\"sparse_newest_center_us\":" << newest_center_us << ",";
        out << "\"sparse_coverage_us\":" << coverage_us << ",";
        out << "\"sparse_total_points_in_ring\":" << total_points << ",";
        out << "\"visual_mask_slice_us\":" << slice_us << ",";
        out << "\"visual_accumulation_us\":" << accumulation_us << ",";
        out << "\"entries\":[";
        bool first = true;
        for (const auto &entry : sparse_entries) {
            if (!entry.valid) {
                continue;
            }
            if (!first) {
                out << ",";
            }
            first = false;
            out << "{";
            out << "\"slot\":" << entry.slot << ",";
            out << "\"shm_name\":\"" << json_escape(entry.shm_name) << "\",";
            out << "\"path\":\"" << json_escape(entry.path) << "\",";
            out << "\"frame_id\":" << entry.frame_id << ",";
            out << "\"event_window_start_us\":" << entry.event_window_start_us << ",";
            out << "\"event_window_end_us\":" << entry.event_window_end_us << ",";
            out << "\"event_window_center_us\":" << entry.event_window_center_us << ",";
            out << "\"published_wall_us\":" << entry.published_wall_us << ",";
            out << "\"sparse_point_count\":" << entry.point_count << ",";
            out << "\"sparse_dropped_point_count\":" << entry.dropped_point_count << ",";
            out << "\"valid\":" << (entry.valid ? "true" : "false");
            out << "}";
        }
        out << "],";
        out << "\"last_error\":\"" << json_escape(last_error) << "\"";
        out << "}\n";
        out.close();
        ::rename(tmp.c_str(), sparse_sidecar_path.c_str());
    }

    void write_sidecar(int64_t latest_ts) {
        if (sidecar_path.empty()) {
            return;
        }
        std::string err;
        if (!ensure_parent_dir(sidecar_path, err)) {
            last_error = err;
            return;
        }
        const std::string tmp = sidecar_path + ".tmp";
        std::ofstream out(tmp);
        if (!out) {
            last_error = "open_sidecar_failed";
            return;
        }
        const bool valid_mask = mask_valid();
        const bool hold_mask = mask_hold_active();
        const bool mask_available = mask_data_available();
        const double mask_age_ms = mask_reader.published_wall_us > 0
            ? static_cast<double>(wall_us() - static_cast<int64_t>(mask_reader.published_wall_us)) / 1000.0
            : -1.0;
        const double reduction = raw_events > 0
            ? static_cast<double>(dropped_events) / static_cast<double>(std::max<uint64_t>(1, raw_events))
            : 0.0;
        out << "{";
        out << "\"schema_version\":1,";
        out << "\"source\":\"hal_visual_mask_overlay\",";
        out << "\"camera\":\"" << json_escape(alias) << "\",";
        out << "\"published_wall_us\":" << wall_us() << ",";
        out << "\"event_ts_us\":" << latest_ts << ",";
        out << "\"overlay_nonzero_pixels\":" << last_nonzero_pixels << ",";
        out << "\"overlay_max_luma\":" << (last_nonzero_pixels > 0 ? 255 : 0) << ",";
        out << "\"visual_mask_filter_enabled\":true,";
        out << "\"visual_mask_slice_us\":" << slice_us << ",";
        out << "\"visual_accumulation_us\":" << accumulation_us << ",";
        out << "\"visual_mask_control_path\":\"" << json_escape(control_path) << "\",";
        out << "\"visual_mask_control_seq\":" << control_seq << ",";
        out << "\"visual_mask_control_apply_count\":" << control_apply_count << ",";
        out << "\"visual_mask_control_error\":\"" << json_escape(control_error) << "\",";
        out << "\"visual_mask_sync_timestamp_policy\":\"event_window_end_us\",";
        out << "\"visual_mask_valid\":" << (valid_mask ? "true" : "false") << ",";
        out << "\"visual_mask_data_available\":" << (mask_available ? "true" : "false") << ",";
        out << "\"visual_mask_hold_active\":" << (hold_mask ? "true" : "false") << ",";
        out << "\"visual_mask_age_ms\":" << std::fixed << std::setprecision(3) << mask_age_ms << ",";
        out << "\"visual_raw_event_count\":" << raw_events << ",";
        out << "\"visual_filtered_event_count\":" << filtered_events << ",";
        out << "\"visual_drop_count\":" << dropped_events << ",";
        out << "\"visual_filter_reduction_ratio\":" << std::fixed << std::setprecision(6) << reduction << ",";
        out << "\"visual_window_event_count\":" << window_events.size() << ",";
        out << "\"visual_publish_count\":" << publish_count << ",";
        out << "\"history_enable\":" << (history_enable ? "true" : "false") << ",";
        out << "\"history_sidecar_path\":\"" << json_escape(history_sidecar_path) << "\",";
        out << "\"history_slots\":" << history_slots << ",";
        out << "\"sparse_enable\":" << (sparse_enable ? "true" : "false") << ",";
        out << "\"sparse_sidecar_path\":\"" << json_escape(sparse_sidecar_path) << "\",";
        out << "\"sparse_slots\":" << sparse_slots << ",";
        out << "\"sparse_max_points\":" << sparse_max_points << ",";
        out << "\"sparse_last_point_count\":" << sparse_last_point_count << ",";
        out << "\"sparse_last_dropped_points\":" << sparse_last_dropped_points << ",";
        out << "\"sparse_write_count\":" << sparse_write_count << ",";
        out << "\"filter_policy\":\"" << json_escape(last_filter_policy) << "\",";
        out << "\"mask_path\":\"" << json_escape(mask_path) << "\",";
        out << "\"mask_seq\":" << mask_reader.last_seq << ",";
        out << "\"mask_read_count\":" << mask_reader.read_count << ",";
        out << "\"mask_skip_count\":" << mask_reader.skip_count << ",";
        out << "\"mask_last_error\":\"" << json_escape(mask_reader.last_error) << "\",";
        out << "\"last_error\":\"" << json_escape(last_error) << "\"";
        out << "}\n";
        out.close();
        ::rename(tmp.c_str(), sidecar_path.c_str());
        write_history_sidecar();
        write_sparse_history_sidecar();
    }

    std::string json_snapshot() const {
        const bool valid_mask = mask_valid();
        const bool hold_mask = mask_hold_active();
        const bool mask_available = mask_data_available();
        const double mask_age_ms = mask_reader.published_wall_us > 0
            ? static_cast<double>(wall_us() - static_cast<int64_t>(mask_reader.published_wall_us)) / 1000.0
            : -1.0;
        const double reduction = raw_events > 0
            ? static_cast<double>(dropped_events) / static_cast<double>(std::max<uint64_t>(1, raw_events))
            : 0.0;
        int valid_history_count = 0;
        int64_t oldest_center_us = 0;
        int64_t newest_center_us = 0;
        for (const auto &entry : history_entries) {
            if (!entry.valid) {
                continue;
            }
            ++valid_history_count;
            if (oldest_center_us == 0 || entry.event_window_center_us < oldest_center_us) {
                oldest_center_us = entry.event_window_center_us;
            }
            if (newest_center_us == 0 || entry.event_window_center_us > newest_center_us) {
                newest_center_us = entry.event_window_center_us;
            }
        }
        const int64_t history_coverage_us = (oldest_center_us > 0 && newest_center_us >= oldest_center_us)
            ? (newest_center_us - oldest_center_us)
            : 0;
        int valid_sparse_count = 0;
        int64_t sparse_oldest_center_us = 0;
        int64_t sparse_newest_center_us = 0;
        for (const auto &entry : sparse_entries) {
            if (!entry.valid) {
                continue;
            }
            ++valid_sparse_count;
            if (sparse_oldest_center_us == 0 || entry.event_window_center_us < sparse_oldest_center_us) {
                sparse_oldest_center_us = entry.event_window_center_us;
            }
            if (sparse_newest_center_us == 0 || entry.event_window_center_us > sparse_newest_center_us) {
                sparse_newest_center_us = entry.event_window_center_us;
            }
        }
        const int64_t sparse_coverage_us = (sparse_oldest_center_us > 0 && sparse_newest_center_us >= sparse_oldest_center_us)
            ? (sparse_newest_center_us - sparse_oldest_center_us)
            : 0;
        std::ostringstream os;
        os << "{";
        os << "\"enabled\":" << (enabled ? "true" : "false") << ",";
        os << "\"overlay_path\":\"" << json_escape(writer.path) << "\",";
        os << "\"sidecar_path\":\"" << json_escape(sidecar_path) << "\",";
        os << "\"mask_path\":\"" << json_escape(mask_path) << "\",";
        os << "\"slice_us\":" << slice_us << ",";
        os << "\"accumulation_us\":" << accumulation_us << ",";
        os << "\"control_path\":\"" << json_escape(control_path) << "\",";
        os << "\"control_seq\":" << control_seq << ",";
        os << "\"control_apply_count\":" << control_apply_count << ",";
        os << "\"control_source\":\"" << json_escape(control_source) << "\",";
        os << "\"control_error\":\"" << json_escape(control_error) << "\",";
        os << "\"sync_timestamp_policy\":\"event_window_end_us\",";
        os << "\"publish_interval_us\":" << publish_interval_us << ",";
        os << "\"no_mask_policy\":\"" << json_escape(no_mask_policy) << "\",";
        os << "\"filter_policy\":\"" << json_escape(last_filter_policy) << "\",";
        os << "\"mask_valid\":" << (valid_mask ? "true" : "false") << ",";
        os << "\"mask_data_available\":" << (mask_available ? "true" : "false") << ",";
        os << "\"mask_hold_active\":" << (hold_mask ? "true" : "false") << ",";
        os << "\"mask_age_ms\":" << std::fixed << std::setprecision(3) << mask_age_ms << ",";
        os << "\"mask_seq\":" << mask_reader.last_seq << ",";
        os << "\"mask_read_count\":" << mask_reader.read_count << ",";
        os << "\"mask_skip_count\":" << mask_reader.skip_count << ",";
        os << "\"mask_last_error\":\"" << json_escape(mask_reader.last_error) << "\",";
        os << "\"raw_event_count\":" << raw_events << ",";
        os << "\"filtered_event_count\":" << filtered_events << ",";
        os << "\"dropped_event_count\":" << dropped_events << ",";
        os << "\"filter_reduction_ratio\":" << std::fixed << std::setprecision(6) << reduction << ",";
        os << "\"window_event_count\":" << window_events.size() << ",";
        os << "\"publish_count\":" << publish_count << ",";
        os << "\"history_enable\":" << (history_enable ? "true" : "false") << ",";
        os << "\"history_sidecar_path\":\"" << json_escape(history_sidecar_path) << "\",";
        os << "\"history_slots\":" << history_slots << ",";
        os << "\"history_cursor\":" << history_cursor << ",";
        os << "\"history_write_count\":" << history_write_count << ",";
        os << "\"history_overwrite_count\":" << history_overwrite_count << ",";
        os << "\"history_valid_count\":" << valid_history_count << ",";
        os << "\"history_coverage_us\":" << history_coverage_us << ",";
        os << "\"sparse_enable\":" << (sparse_enable ? "true" : "false") << ",";
        os << "\"sparse_sidecar_path\":\"" << json_escape(sparse_sidecar_path) << "\",";
        os << "\"sparse_slots\":" << sparse_slots << ",";
        os << "\"sparse_cursor\":" << sparse_cursor << ",";
        os << "\"sparse_max_points\":" << sparse_max_points << ",";
        os << "\"sparse_write_count\":" << sparse_write_count << ",";
        os << "\"sparse_overwrite_count\":" << sparse_overwrite_count << ",";
        os << "\"sparse_valid_count\":" << valid_sparse_count << ",";
        os << "\"sparse_coverage_us\":" << sparse_coverage_us << ",";
        os << "\"sparse_last_point_count\":" << sparse_last_point_count << ",";
        os << "\"sparse_last_dropped_points\":" << sparse_last_dropped_points << ",";
        os << "\"sparse_total_dropped_points\":" << sparse_total_dropped_points << ",";
        os << "\"last_nonzero_pixels\":" << last_nonzero_pixels << ",";
        os << "\"last_error\":\"" << json_escape(last_error) << "\"";
        os << "}";
        return os.str();
    }
};

struct CameraRuntime {
    std::unique_ptr<Metavision::Device> device;
    Metavision::I_EventsStream *stream = nullptr;
    Metavision::I_EventsStreamDecoder *decoder = nullptr;
    Metavision::I_EventDecoder<Metavision::EventCD> *cd = nullptr;
    Metavision::I_EventDecoder<Metavision::EventExtTrigger> *trigger_decoder = nullptr;
    std::unique_ptr<CameraStats> stats;
    FrameWriter writer;
    SliceEmitter emitter;
    VisualMaskOverlay visual_mask_overlay;
    std::thread thread;
    uint64_t raw_record_last_seq = 0;
    int64_t raw_record_last_poll_us = 0;
};

std::string raw_record_path_for(const Options &opt, const CameraStats &st, const std::string &session_id) {
    std::string err;
    if (!ensure_dir(opt.raw_record_dir, err)) {
        throw std::runtime_error("raw_record_dir: " + err);
    }
    const std::string session_dir = join_path(opt.raw_record_dir, sanitize_filename(session_id));
    if (!ensure_dir(session_dir, err)) {
        throw std::runtime_error("raw_record_session_dir: " + err);
    }
    const std::string prefix = sanitize_filename(opt.raw_record_prefix);
    const std::string alias = sanitize_filename(st.alias);
    const std::string serial = sanitize_filename(st.requested_serial.empty() ? st.opened_serial : st.requested_serial);
    std::string filename;
    if (!prefix.empty()) {
        filename += prefix + "_";
    }
    filename += "EVENT_" + alias + "_" + serial + ".raw";
    return join_path(session_dir, filename);
}

bool broker_start_raw_recording(CameraRuntime *rt, const Options *opt, const std::string &session_id_raw,
                                const std::string &cmd_name) {
    if (!rt || !rt->stream || !opt) {
        return false;
    }
    auto &st = *rt->stats;
    const std::string session_id = session_id_raw.empty() ? now_session_id() : session_id_raw;
    try {
        std::string current_path;
        {
            std::lock_guard<std::mutex> lk(st.mu);
            current_path = st.raw_record_path;
        }
        if (st.raw_recording.load(std::memory_order_relaxed)) {
            st.set_raw_record_state(true, current_path, session_id, cmd_name);
            return true;
        }
        const std::string path = raw_record_path_for(*opt, st, session_id);
        if (!rt->stream->log_raw_data(path)) {
            st.set_raw_record_error("I_EventsStream::log_raw_data returned false: " + path);
            return false;
        }
        st.raw_record_start_count.fetch_add(1, std::memory_order_relaxed);
        st.set_raw_record_state(true, path, session_id, cmd_name);
        std::cout << "[raw_record] started alias=" << st.alias << " path=" << path << "\n";
        return true;
    } catch (const std::exception &e) {
        st.set_raw_record_error(std::string("raw record start failed: ") + e.what());
        return false;
    }
}

bool broker_stop_raw_recording(CameraRuntime *rt, const std::string &cmd_name) {
    if (!rt || !rt->stream) {
        return false;
    }
    auto &st = *rt->stats;
    if (!st.raw_recording.load(std::memory_order_relaxed)) {
        std::string path;
        std::string session;
        {
            std::lock_guard<std::mutex> lk(st.mu);
            path = st.raw_record_path;
            session = st.raw_record_session_id;
        }
        st.set_raw_record_state(false, path, session, cmd_name);
        return false;
    }
    std::string path;
    std::string session;
    {
        std::lock_guard<std::mutex> lk(st.mu);
        path = st.raw_record_path;
        session = st.raw_record_session_id;
    }
    try {
        rt->stream->stop_log_raw_data();
        st.raw_record_stop_count.fetch_add(1, std::memory_order_relaxed);
        st.set_raw_record_state(false, path, session, cmd_name);
        std::cout << "[raw_record] stopped alias=" << st.alias << " path=" << path << "\n";
        return true;
    } catch (const std::exception &e) {
        st.set_raw_record_error(std::string("raw record stop failed: ") + e.what());
        return false;
    }
}

void broker_poll_raw_record_command(CameraRuntime *rt, const Options *opt) {
    if (!rt || !opt || opt->raw_record_command_file.empty()) {
        return;
    }
    auto &st = *rt->stats;
    st.raw_record_enabled.store(true, std::memory_order_relaxed);
    const int64_t now = steady_us();
    const int64_t poll_us = static_cast<int64_t>(std::max(5, opt->raw_record_poll_ms)) * 1000;
    if (rt->raw_record_last_poll_us > 0 && now - rt->raw_record_last_poll_us < poll_us) {
        return;
    }
    rt->raw_record_last_poll_us = now;
    const std::string text = read_text_file(opt->raw_record_command_file);
    if (text.empty()) {
        return;
    }
    const uint64_t seq = json_uint_value(text, "seq");
    if (seq == 0 || seq == rt->raw_record_last_seq) {
        return;
    }
    rt->raw_record_last_seq = seq;
    st.raw_record_last_seq.store(seq, std::memory_order_relaxed);
    if (!raw_record_targets_event(text)) {
        return;
    }
    st.raw_record_commands_seen.fetch_add(1, std::memory_order_relaxed);
    std::string cmd = json_string_value(text, "cmd");
    if (cmd.empty()) {
        cmd = json_string_value(text, "command");
    }
    std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    const std::string session_id = json_string_value(text, "session_id");
    if (cmd == "start") {
        broker_start_raw_recording(rt, opt, session_id, cmd);
    } else if (cmd == "stop") {
        broker_stop_raw_recording(rt, cmd);
    } else if (cmd == "toggle") {
        if (st.raw_recording.load(std::memory_order_relaxed)) {
            broker_stop_raw_recording(rt, cmd);
        } else {
            broker_start_raw_recording(rt, opt, session_id, cmd);
        }
    } else if (!cmd.empty()) {
        st.set_raw_record_error("unknown raw record command: " + cmd);
    }
}

void apply_optional_config(Metavision::Device &device, CameraStats &st, const Options &opt) {
    std::ostringstream log;

    auto *geo = device.get_facility<Metavision::I_Geometry>();
    if (geo) {
        log << "geometry=" << geo->get_width() << "x" << geo->get_height() << ";";
    } else {
        log << "geometry=missing;";
    }

    auto *plugin = device.get_facility<Metavision::I_PluginSoftwareInfo>();
    if (plugin) {
        log << "plugin=" << plugin->get_plugin_name() << ";";
    } else {
        log << "plugin=missing;";
    }

    auto *hw = device.get_facility<Metavision::I_HW_Identification>();
    if (hw) {
        try {
            log << "system_id=" << hw->get_system_id() << ";";
        } catch (...) {
            log << "system_id=err;";
        }
    } else {
        log << "hw_id=missing;";
    }

    if (opt.enable_trigger_in) {
        st.trigger_in_requested.store(true, std::memory_order_relaxed);
        st.trigger_in_enable_attempted.store(true, std::memory_order_relaxed);
        auto *trigger = device.get_facility<Metavision::I_TriggerIn>();
        if (trigger) {
            const bool ok = trigger->enable(Metavision::I_TriggerIn::Channel::Main);
            st.trigger_in_enable_ok.store(ok, std::memory_order_relaxed);
            log << "trigger_in_main=" << (ok ? "ok" : "failed") << ";";
        } else {
            log << "trigger_in=missing;";
        }
    } else {
        st.trigger_in_requested.store(false, std::memory_order_relaxed);
    }

    if (opt.enable_erc) {
        auto *erc = device.get_facility<Metavision::I_ErcModule>();
        if (erc) {
            const bool en = erc->enable(true);
            const bool sr = erc->set_cd_event_rate(opt.erc_rate);
            log << "erc_enable=" << (en ? "ok" : "failed") << ";erc_rate=" << (sr ? "ok" : "failed") << ";";
        } else {
            log << "erc=missing;";
        }
    }

    if (opt.enable_trail) {
        auto *trail = device.get_facility<Metavision::I_EventTrailFilterModule>();
        if (trail) {
            const auto type = parse_trail_type(opt.trail_type);
            const bool ty   = trail->set_type(type);
            const bool th   = trail->set_threshold(opt.trail_th_us);
            const bool en   = trail->enable(true);
            log << "trail_type=" << opt.trail_type << ":" << (ty ? "ok" : "failed")
                << ";trail_th=" << (th ? "ok" : "failed") << ";trail_enable=" << (en ? "ok" : "failed") << ";";
        } else {
            log << "trail=missing;";
        }
    }

    if (!opt.biases.empty()) {
        auto *biases = device.get_facility<Metavision::I_LL_Biases>();
        if (biases) {
            for (const auto &kv : opt.biases) {
                try {
                    const bool ok = biases->set(kv.first, kv.second);
                    log << "bias_" << kv.first << "=" << kv.second << ":" << (ok ? "ok" : "failed") << ";";
                } catch (const std::exception &e) {
                    log << "bias_" << kv.first << "=" << kv.second << ":err:" << e.what() << ";";
                }
            }
        } else {
            log << "biases=missing;";
        }
    }

    st.config_log = log.str();
}

std::unique_ptr<CameraRuntime> open_runtime(const std::string &alias, const std::string &serial, const Options &opt) {
    auto rt                     = std::make_unique<CameraRuntime>();
    rt->stats                   = std::make_unique<CameraStats>();
    rt->stats->alias            = alias;
    rt->stats->requested_serial = serial;

    auto candidates = serial_candidates(serial);
    std::ostringstream attempts;
    for (int attempt = 1; attempt <= opt.open_retries; ++attempt) {
        for (const auto &candidate : candidates) {
            attempts << "[" << candidate << "@" << attempt << "]";
            try {
                Metavision::DeviceConfig cfg;
                cfg.enable_biases_range_check_bypass(opt.bypass_bias_range);
                rt->device = Metavision::DeviceDiscovery::open(candidate, cfg);
                if (!rt->device) {
                    attempts << ":null;";
                    continue;
                }
                rt->stats->opened_serial = candidate.empty() ? "first_available" : candidate;
                rt->stats->opened.store(true, std::memory_order_relaxed);
                break;
            } catch (const Metavision::BaseException &e) {
                attempts << ":base_exception:" << e.what() << ";";
                rt->stats->open_error = e.what();
            } catch (const std::exception &e) {
                attempts << ":exception:" << e.what() << ";";
                rt->stats->open_error = e.what();
            }
        }
        if (rt->device || attempt >= opt.open_retries) {
            break;
        }
        if (opt.open_retry_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(opt.open_retry_delay_ms));
        }
    }
    rt->stats->open_attempts = attempts.str();
    if (!rt->device) {
        if (rt->stats->open_error.empty()) {
            rt->stats->open_error = "DeviceDiscovery::open returned null for all candidates";
        }
        return rt;
    }

    apply_optional_config(*rt->device, *rt->stats, opt);

    auto *geo = rt->device->get_facility<Metavision::I_Geometry>();
    rt->stream  = rt->device->get_facility<Metavision::I_EventsStream>();
    rt->decoder = rt->device->get_facility<Metavision::I_EventsStreamDecoder>();
    rt->cd      = rt->device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();
    rt->trigger_decoder = rt->device->get_facility<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
    rt->stats->ext_trigger_decoder_available.store(rt->trigger_decoder != nullptr, std::memory_order_relaxed);
    if (!rt->stream || !rt->decoder || !rt->cd || !geo) {
        std::ostringstream err;
        err << "missing facilities stream=" << (rt->stream ? "ok" : "missing")
            << " decoder=" << (rt->decoder ? "ok" : "missing")
            << " cd=" << (rt->cd ? "ok" : "missing")
            << " geometry=" << (geo ? "ok" : "missing");
        rt->stats->open_error = err.str();
        return rt;
    }
    rt->stats->raw_record_enabled.store(!opt.raw_record_command_file.empty(), std::memory_order_relaxed);

    if (!rt->writer.open_for_camera(opt, alias, *rt->stats)) {
        return rt;
    }

    rt->emitter.width = static_cast<uint32_t>(geo->get_width());
    rt->emitter.height = static_cast<uint32_t>(geo->get_height());
    rt->emitter.slice_us = opt.slice_us;
    rt->emitter.writer = &rt->writer;

    {
        std::string err;
        if (!rt->visual_mask_overlay.init(opt, alias, rt->emitter.width, rt->emitter.height, err)) {
            rt->stats->set_last_error("visual_mask_overlay: " + err);
            return rt;
        }
    }

    auto *rt_ptr = rt.get();
    rt->cd->add_event_buffer_callback([rt_ptr](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        auto &st = *rt_ptr->stats;
        st.callbacks.fetch_add(1, std::memory_order_relaxed);
        const uint64_t n = static_cast<uint64_t>(std::distance(begin, end));
        st.events_seen.fetch_add(n, std::memory_order_relaxed);
        if (n == 0) {
            st.callback_empty.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        st.on_ts(begin->t);
        st.on_ts((end - 1)->t);
        if (!rt_ptr->emitter.on_events(begin, end)) {
            g_stop.store(true, std::memory_order_relaxed);
        }
        rt_ptr->visual_mask_overlay.on_events(begin, end);
    });

    if (rt->trigger_decoder) {
        rt->trigger_decoder->add_event_buffer_callback(
            [rt_ptr](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                auto &st = *rt_ptr->stats;
                st.trigger_callbacks.fetch_add(1, std::memory_order_relaxed);
                const uint64_t n = static_cast<uint64_t>(std::distance(begin, end));
                st.triggers_seen.fetch_add(n, std::memory_order_relaxed);
                if (n > 0) {
                    st.on_ext_trigger_ts(begin->t);
                    st.on_ext_trigger_ts((end - 1)->t);
                    rt_ptr->emitter.on_triggers(begin, end);
                }
            });
    }

    return rt;
}

void decode_loop(CameraRuntime *rt, const Options *opt) {
    auto &st = *rt->stats;
    try {
        rt->emitter.emit_hello();
        rt->stream->start();
        st.started.store(true, std::memory_order_relaxed);
        while (!g_stop.load(std::memory_order_relaxed)) {
            broker_poll_raw_record_command(rt, opt);
            rt->visual_mask_overlay.poll_control();
            short ret = 0;
            if (opt->wait_mode == "blocking") {
                ret = rt->stream->wait_next_buffer();
            } else {
                ret = rt->stream->poll_buffer();
            }

            if (ret < 0) {
                st.poll_neg.fetch_add(1, std::memory_order_relaxed);
                break;
            }
            if (ret == 0) {
                st.poll_zero.fetch_add(1, std::memory_order_relaxed);
                if (opt->poll_sleep_us > 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(opt->poll_sleep_us));
                }
                continue;
            }
            st.poll_one.fetch_add(1, std::memory_order_relaxed);

            auto raw_data = rt->stream->get_latest_raw_data();
            if (!raw_data) {
                continue;
            }
            st.raw_buffers.fetch_add(1, std::memory_order_relaxed);
            st.raw_bytes.fetch_add(static_cast<uint64_t>(raw_data->size()), std::memory_order_relaxed);
            const int64_t t0 = steady_us();
            rt->decoder->decode(raw_data->data(), raw_data->data() + raw_data->size());
            const int64_t t1 = steady_us();
            st.decode_calls.fetch_add(1, std::memory_order_relaxed);
            st.decode_us.fetch_add(static_cast<uint64_t>(std::max<int64_t>(0, t1 - t0)), std::memory_order_relaxed);
        }
        broker_stop_raw_recording(rt, "shutdown");
        try {
            rt->stream->stop();
        } catch (...) {
        }
        st.stream_stopped.store(true, std::memory_order_relaxed);
    } catch (const std::exception &e) {
        st.set_last_error(e.what());
        broker_stop_raw_recording(rt, "exception");
        try {
            if (rt->stream) {
                rt->stream->stop();
            }
        } catch (...) {
        }
        st.stream_stopped.store(true, std::memory_order_relaxed);
    }
}

Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto value = [&](const std::string &name) {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--serials") {
            opt.serials_csv = value(arg);
        } else if (arg == "--aliases") {
            opt.aliases_csv = value(arg);
        } else if (arg == "--output-mode") {
            opt.output_mode = value(arg);
        } else if (arg == "--out-dir") {
            opt.out_dir = value(arg);
        } else if (arg == "--fifo-open-mode") {
            opt.fifo_open_mode = value(arg);
        } else if (arg == "--stats-jsonl") {
            opt.stats_jsonl = value(arg);
        } else if (arg == "--duration-sec") {
            opt.duration_sec = std::stoi(value(arg));
        } else if (arg == "--slice-us") {
            opt.slice_us = std::stoi(value(arg));
        } else if (arg == "--report-ms") {
            opt.report_ms = std::stoi(value(arg));
        } else if (arg == "--wait-mode") {
            opt.wait_mode = value(arg);
        } else if (arg == "--poll-sleep-us") {
            opt.poll_sleep_us = std::stoi(value(arg));
        } else if (arg == "--enable-trigger-in") {
            opt.enable_trigger_in = true;
        } else if (arg == "--enable-erc") {
            opt.enable_erc = true;
        } else if (arg == "--erc-rate") {
            opt.erc_rate = static_cast<uint32_t>(std::stoul(value(arg)));
        } else if (arg == "--enable-trail") {
            opt.enable_trail = true;
        } else if (arg == "--trail-type") {
            opt.trail_type = value(arg);
        } else if (arg == "--trail-threshold-us") {
            opt.trail_th_us = static_cast<uint32_t>(std::stoul(value(arg)));
        } else if (arg == "--set-bias") {
            const std::string s = value(arg);
            const auto pos = s.find('=');
            if (pos == std::string::npos) {
                throw std::runtime_error("--set-bias expects name=value");
            }
            opt.biases.emplace_back(s.substr(0, pos), std::stoi(s.substr(pos + 1)));
        } else if (arg == "--no-bias-range-bypass") {
            opt.bypass_bias_range = false;
        } else if (arg == "--open-retries") {
            opt.open_retries = std::stoi(value(arg));
        } else if (arg == "--open-retry-delay-ms") {
            opt.open_retry_delay_ms = std::stoi(value(arg));
        } else if (arg == "--raw-record-command-file") {
            opt.raw_record_command_file = value(arg);
        } else if (arg == "--raw-record-dir") {
            opt.raw_record_dir = value(arg);
        } else if (arg == "--raw-record-prefix") {
            opt.raw_record_prefix = value(arg);
        } else if (arg == "--raw-record-poll-ms") {
            opt.raw_record_poll_ms = std::stoi(value(arg));
        } else if (arg == "--visual-mask-overlay-enable") {
            opt.visual_mask_overlay_enable = true;
        } else if (arg == "--disable-visual-mask-overlay") {
            opt.visual_mask_overlay_enable = false;
        } else if (arg == "--visual-mask-raster-template") {
            opt.visual_mask_raster_template = value(arg);
        } else if (arg == "--visual-mask-overlay-name-template") {
            opt.visual_mask_overlay_name_template = value(arg);
        } else if (arg == "--visual-mask-overlay-sidecar-template") {
            opt.visual_mask_overlay_sidecar_template = value(arg);
        } else if (arg == "--visual-mask-overlay-history-enable") {
            opt.visual_mask_overlay_history_enable = true;
        } else if (arg == "--disable-visual-mask-overlay-history") {
            opt.visual_mask_overlay_history_enable = false;
        } else if (arg == "--visual-mask-overlay-history-name-template") {
            opt.visual_mask_overlay_history_name_template = value(arg);
        } else if (arg == "--visual-mask-overlay-history-sidecar-template") {
            opt.visual_mask_overlay_history_sidecar_template = value(arg);
        } else if (arg == "--visual-mask-overlay-history-slots") {
            opt.visual_mask_overlay_history_slots = std::stoi(value(arg));
        } else if (arg == "--visual-mask-overlay-sparse-enable") {
            opt.visual_mask_overlay_sparse_enable = true;
        } else if (arg == "--disable-visual-mask-overlay-sparse") {
            opt.visual_mask_overlay_sparse_enable = false;
        } else if (arg == "--visual-mask-overlay-sparse-name-template") {
            opt.visual_mask_overlay_sparse_name_template = value(arg);
        } else if (arg == "--visual-mask-overlay-sparse-sidecar-template") {
            opt.visual_mask_overlay_sparse_sidecar_template = value(arg);
        } else if (arg == "--visual-mask-overlay-sparse-slots") {
            opt.visual_mask_overlay_sparse_slots = std::stoi(value(arg));
        } else if (arg == "--visual-mask-overlay-sparse-max-points") {
            opt.visual_mask_overlay_sparse_max_points = std::stoi(value(arg));
        } else if (arg == "--visual-mask-no-mask-policy") {
            opt.visual_mask_no_mask_policy = value(arg);
        } else if (arg == "--visual-mask-slice-us") {
            opt.visual_mask_slice_us = std::stoi(value(arg));
        } else if (arg == "--visual-mask-accumulation-us") {
            opt.visual_mask_accumulation_us = std::stoi(value(arg));
        } else if (arg == "--visual-mask-publish-fps") {
            opt.visual_mask_publish_fps = std::stoi(value(arg));
        } else if (arg == "--visual-mask-stale-ms") {
            opt.visual_mask_stale_ms = std::stoi(value(arg));
        } else if (arg == "--visual-mask-control-json") {
            opt.visual_mask_control_json = value(arg);
        } else if (arg == "--visual-mask-control-poll-ms") {
            opt.visual_mask_control_poll_ms = std::stoi(value(arg));
        } else if (arg == "--list-only") {
            opt.list_only = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: hal_event_fifo_broker --serials s1,s2 --aliases A,B "
                         "[--output-mode dry-run|fifo --out-dir dir --duration-sec 0] "
                         "[--raw-record-command-file sync_ipc/record_control.json --raw-record-dir raw_records] "
                         "[--visual-mask-overlay-enable --visual-mask-slice-us 33333 --visual-mask-accumulation-us 33333] "
                         "[--visual-mask-control-json sync_ipc/event_mask_control.json] "
                         "[--visual-mask-overlay-sparse-enable --visual-mask-overlay-sparse-max-points 20000]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.wait_mode != "blocking" && opt.wait_mode != "poll") {
        throw std::runtime_error("--wait-mode must be blocking or poll");
    }
    if (opt.output_mode != "dry-run" && opt.output_mode != "fifo") {
        throw std::runtime_error("--output-mode must be dry-run or fifo");
    }
    if (opt.fifo_open_mode != "rdwr" && opt.fifo_open_mode != "blocking" && opt.fifo_open_mode != "nonblock") {
        throw std::runtime_error("--fifo-open-mode must be rdwr, blocking, or nonblock");
    }
    if (opt.output_mode == "fifo" && opt.out_dir.empty()) {
        throw std::runtime_error("--out-dir is required for --output-mode fifo");
    }
    opt.duration_sec = std::max(0, opt.duration_sec);
    opt.slice_us = std::max(1, opt.slice_us);
    opt.report_ms = std::max(100, opt.report_ms);
    opt.poll_sleep_us = std::max(0, opt.poll_sleep_us);
    opt.open_retries = std::max(1, opt.open_retries);
    opt.open_retry_delay_ms = std::max(0, opt.open_retry_delay_ms);
    opt.raw_record_poll_ms = std::max(5, opt.raw_record_poll_ms);
    opt.visual_mask_slice_us = std::max(1000, opt.visual_mask_slice_us);
    opt.visual_mask_accumulation_us = std::max(opt.visual_mask_slice_us, opt.visual_mask_accumulation_us);
    opt.visual_mask_publish_fps = std::max(1, opt.visual_mask_publish_fps);
    opt.visual_mask_stale_ms = std::max(1, opt.visual_mask_stale_ms);
    opt.visual_mask_control_poll_ms = std::max(50, opt.visual_mask_control_poll_ms);
    opt.visual_mask_overlay_history_slots = std::max(1, std::min(256, opt.visual_mask_overlay_history_slots));
    return opt;
}

std::string aggregate_json(const std::vector<std::unique_ptr<CameraRuntime>> &runtimes, uint64_t tick,
                           double elapsed_s, double cpu_total_s, double cpu_pct_delta, double cpu_pct_total,
                           const Options &opt) {
    uint64_t opened = 0, started = 0, events = 0, triggers = 0, frames = 0, raw_buffers = 0, raw_bytes = 0;
    uint64_t write_errors = 0;
    uint64_t raw_recording_cameras = 0, raw_record_commands_seen = 0, raw_record_errors = 0;
    for (const auto &rt : runtimes) {
        const auto &st = *rt->stats;
        opened += st.opened.load(std::memory_order_relaxed) ? 1 : 0;
        started += st.started.load(std::memory_order_relaxed) ? 1 : 0;
        events += st.events_seen.load(std::memory_order_relaxed);
        triggers += st.triggers_seen.load(std::memory_order_relaxed);
        frames += st.frames_written.load(std::memory_order_relaxed);
        raw_buffers += st.raw_buffers.load(std::memory_order_relaxed);
        raw_bytes += st.raw_bytes.load(std::memory_order_relaxed);
        write_errors += st.write_errors.load(std::memory_order_relaxed);
        raw_recording_cameras += st.raw_recording.load(std::memory_order_relaxed) ? 1 : 0;
        raw_record_commands_seen += st.raw_record_commands_seen.load(std::memory_order_relaxed);
        raw_record_errors += st.raw_record_error_count.load(std::memory_order_relaxed);
    }

    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "{";
    os << "\"tick\":" << tick << ",";
    os << "\"output_mode\":\"" << opt.output_mode << "\",";
    os << "\"wait_mode\":\"" << opt.wait_mode << "\",";
    os << "\"slice_us\":" << opt.slice_us << ",";
    os << "\"elapsed_s\":" << elapsed_s << ",";
    os << "\"cpu_total_s\":" << cpu_total_s << ",";
    os << "\"cpu_pct_delta\":" << cpu_pct_delta << ",";
    os << "\"cpu_pct_total\":" << cpu_pct_total << ",";
    os << "\"cameras_requested\":" << runtimes.size() << ",";
    os << "\"cameras_opened\":" << opened << ",";
    os << "\"cameras_started\":" << started << ",";
    os << "\"events_seen\":" << events << ",";
    os << "\"triggers_seen\":" << triggers << ",";
    os << "\"frames_written\":" << frames << ",";
    os << "\"raw_buffers\":" << raw_buffers << ",";
    os << "\"raw_bytes\":" << raw_bytes << ",";
    os << "\"write_errors\":" << write_errors << ",";
    os << "\"raw_record_command_file\":\"" << json_escape(opt.raw_record_command_file) << "\",";
    os << "\"raw_record_dir\":\"" << json_escape(opt.raw_record_dir) << "\",";
    os << "\"raw_recording_cameras\":" << raw_recording_cameras << ",";
    os << "\"raw_record_commands_seen\":" << raw_record_commands_seen << ",";
    os << "\"raw_record_errors\":" << raw_record_errors << ",";
    os << "\"visual_mask_overlay_enable\":" << (opt.visual_mask_overlay_enable ? "true" : "false") << ",";
    os << "\"visual_mask_slice_us\":" << opt.visual_mask_slice_us << ",";
    os << "\"visual_mask_accumulation_us\":" << opt.visual_mask_accumulation_us << ",";
    os << "\"visual_mask_publish_fps\":" << opt.visual_mask_publish_fps << ",";
    os << "\"visual_mask_control_json\":\"" << json_escape(opt.visual_mask_control_json) << "\",";
    os << "\"visual_mask_control_poll_ms\":" << opt.visual_mask_control_poll_ms << ",";
    os << "\"visual_mask_overlay_history_enable\":" << (opt.visual_mask_overlay_history_enable ? "true" : "false") << ",";
    os << "\"visual_mask_overlay_history_slots\":" << opt.visual_mask_overlay_history_slots << ",";
    os << "\"visual_mask_no_mask_policy\":\"" << json_escape(opt.visual_mask_no_mask_policy) << "\",";
    os << "\"cameras\":[";
    for (size_t i = 0; i < runtimes.size(); ++i) {
        if (i) {
            os << ",";
        }
        std::string cam_json = runtimes[i]->stats->json_snapshot();
        if (!cam_json.empty() && cam_json.back() == '}') {
            cam_json.pop_back();
            cam_json += ",\"visual_mask_overlay\":";
            cam_json += runtimes[i]->visual_mask_overlay.json_snapshot();
            cam_json += "}";
        }
        os << cam_json;
    }
    os << "]}";
    return os.str();
}

} // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        const Options opt = parse_args(argc, argv);
        append_source_report(std::cout);
        if (opt.list_only) {
            return 0;
        }

        auto serials = split_csv(opt.serials_csv);
        auto aliases = split_csv(opt.aliases_csv);
        if (serials.empty()) {
            serials.push_back("");
        }
        while (aliases.size() < serials.size()) {
            aliases.push_back("cam" + std::to_string(aliases.size()));
        }

        std::vector<std::unique_ptr<CameraRuntime>> runtimes;
        runtimes.reserve(serials.size());
        for (size_t i = 0; i < serials.size(); ++i) {
            std::cout << "opening alias=" << aliases[i] << " serial=" << serials[i] << "\n";
            auto rt = open_runtime(aliases[i], serials[i], opt);
            std::cout << "open_result alias=" << aliases[i]
                      << " opened=" << (rt->stats->opened.load() ? "true" : "false")
                      << " opened_serial=" << rt->stats->opened_serial
                      << " fifo_opened=" << (rt->stats->fifo_opened.load() ? "true" : "false")
                      << " error=" << rt->stats->open_error
                      << " attempts=" << rt->stats->open_attempts
                      << " config=" << rt->stats->config_log << "\n";
            runtimes.push_back(std::move(rt));
        }

        std::string stats_dir_err;
        if (!opt.stats_jsonl.empty() && !ensure_parent_dir(opt.stats_jsonl, stats_dir_err)) {
            std::cerr << "WARN: failed to create stats jsonl parent: " << opt.stats_jsonl
                      << " error=" << stats_dir_err << "\n";
        }
        std::ofstream jsonl;
        std::string active_stats_jsonl = opt.stats_jsonl;
        if (!active_stats_jsonl.empty()) {
            jsonl.open(active_stats_jsonl, std::ios::out | std::ios::trunc);
            if (!jsonl) {
                const std::string fallback =
                    active_stats_jsonl + "." + std::to_string(static_cast<long long>(::getpid())) + ".jsonl";
                std::cerr << "WARN: failed to open stats jsonl: " << active_stats_jsonl
                          << "; trying fallback=" << fallback << "\n";
                jsonl.clear();
                active_stats_jsonl = fallback;
                jsonl.open(active_stats_jsonl, std::ios::out | std::ios::trunc);
            }
            if (!jsonl) {
                std::cerr << "WARN: failed to open fallback stats jsonl: " << active_stats_jsonl
                          << "; continuing without broker stats jsonl\n";
                jsonl.clear();
                active_stats_jsonl.clear();
            } else {
                std::cout << "stats_jsonl=" << active_stats_jsonl << "\n";
            }
        }

        size_t started_candidates = 0;
        for (auto &rt : runtimes) {
            if (rt->stats->opened.load() && rt->stream && rt->decoder && rt->cd) {
                if (opt.output_mode == "fifo" && !rt->stats->fifo_opened.load()) {
                    continue;
                }
                ++started_candidates;
                rt->thread = std::thread(decode_loop, rt.get(), &opt);
            }
        }

        const int64_t start_us = steady_us();
        int64_t last_us = start_us;
        const double cpu_start = process_cpu_seconds();
        double cpu_last = cpu_start;
        uint64_t tick = 0;

        if (started_candidates == 0) {
            const auto line = aggregate_json(runtimes, tick, 0.0, 0.0, 0.0, 0.0, opt);
            if (jsonl) {
                jsonl << line << "\n";
                jsonl.close();
            }
            std::cout << "final=" << line << "\n";
            return 2;
        }

        while (!g_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(opt.report_ms));
            ++tick;
            const int64_t now_us = steady_us();
            const double cpu_now = process_cpu_seconds();
            const double elapsed_s = static_cast<double>(now_us - start_us) / 1e6;
            const double delta_wall_s = std::max(1e-6, static_cast<double>(now_us - last_us) / 1e6);
            const double delta_cpu_s = cpu_now - cpu_last;
            const double cpu_delta = 100.0 * delta_cpu_s / delta_wall_s;
            const double cpu_total = 100.0 * (cpu_now - cpu_start) / std::max(1e-6, elapsed_s);
            const auto line = aggregate_json(runtimes, tick, elapsed_s, cpu_now - cpu_start, cpu_delta, cpu_total, opt);
            if (jsonl) {
                jsonl << line << "\n";
                jsonl.flush();
            }
            std::cout << line << "\n";
            last_us = now_us;
            cpu_last = cpu_now;
            if (opt.duration_sec > 0 && elapsed_s >= opt.duration_sec) {
                break;
            }
        }

        g_stop.store(true, std::memory_order_relaxed);
        for (auto &rt : runtimes) {
            if (rt->stream && rt->stats->started.load()) {
                try {
                    rt->stream->stop();
                } catch (...) {
                }
            }
        }
        for (auto &rt : runtimes) {
            if (rt->thread.joinable()) {
                rt->thread.join();
            }
            rt->emitter.finalize();
            rt->visual_mask_overlay.writer.close();
            for (auto &hw : rt->visual_mask_overlay.history_writers) {
                hw.close();
            }
            rt->visual_mask_overlay.mask_reader.close();
            rt->writer.close();
        }

        const int64_t end_us = steady_us();
        const double elapsed_s = static_cast<double>(end_us - start_us) / 1e6;
        const double cpu_total_s = process_cpu_seconds() - cpu_start;
        const double cpu_pct_total = 100.0 * cpu_total_s / std::max(1e-6, elapsed_s);
        const auto final_line = aggregate_json(runtimes, tick + 1, elapsed_s, cpu_total_s, 0.0, cpu_pct_total, opt);
        if (jsonl) {
            jsonl << final_line << "\n";
            jsonl.close();
        }
        std::cout << "final=" << final_line << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

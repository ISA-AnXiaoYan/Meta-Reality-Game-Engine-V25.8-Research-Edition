#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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
#include <metavision/hal/facilities/i_ll_biases.h>
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

struct Options {
    std::string serial;
    int slice_us          = 4000;
    int poll_sleep_us     = 100;
    std::string wait_mode = "poll";
    bool enable_trigger_in = false;
    bool enable_erc        = false;
    uint32_t erc_rate      = 10000000;
    bool enable_trail      = false;
    std::string trail_type = "stc_keep_trail";
    uint32_t trail_th_us   = 10000;
    bool bypass_bias_range = true;
    int open_retries       = 15;
    int open_retry_delay_ms = 1500;
    std::vector<std::pair<std::string, int>> biases;
};

int64_t steady_us() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
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

void write_frame(uint32_t width, uint32_t height, uint64_t seq, uint32_t flags, int64_t slice_start_us,
                 int64_t slice_end_us, int64_t current_time_us, const std::vector<WireEventCD> &events,
                 const std::vector<WireExtTrigger> &triggers) {
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

    if (std::fwrite(&h, 1, sizeof(h), stdout) != sizeof(h)) {
        g_stop.store(true);
        return;
    }
    if (!events.empty()) {
        const size_t n = events.size() * sizeof(WireEventCD);
        if (std::fwrite(events.data(), 1, n, stdout) != n) {
            g_stop.store(true);
            return;
        }
    }
    if (!triggers.empty()) {
        const size_t n = triggers.size() * sizeof(WireExtTrigger);
        if (std::fwrite(triggers.data(), 1, n, stdout) != n) {
            g_stop.store(true);
            return;
        }
    }
    std::fflush(stdout);
}

struct SliceEmitter {
    uint32_t width = 0;
    uint32_t height = 0;
    int slice_us = 4000;
    uint64_t seq = 0;
    int64_t current_idx = -1;
    std::vector<WireEventCD> current_events;
    std::map<int64_t, std::vector<WireExtTrigger>> pending_triggers;

    void emit_hello() {
        const std::vector<WireEventCD> empty_events;
        const std::vector<WireExtTrigger> empty_triggers;
        write_frame(width, height, seq++, kFlagHello, 0, 0, 0, empty_events, empty_triggers);
    }

    void emit_idx(int64_t idx) {
        auto trig_it = pending_triggers.find(idx);
        std::vector<WireExtTrigger> triggers;
        if (trig_it != pending_triggers.end()) {
            triggers.swap(trig_it->second);
            pending_triggers.erase(trig_it);
        }
        const int64_t start_us = idx * static_cast<int64_t>(slice_us);
        const int64_t end_us = start_us + static_cast<int64_t>(slice_us);
        write_frame(width, height, seq++, 0, start_us, end_us, end_us, current_events, triggers);
        current_events.clear();
    }

    void advance_to(int64_t idx) {
        if (current_idx < 0) {
            current_idx = idx;
            return;
        }
        while (current_idx < idx) {
            emit_idx(current_idx);
            ++current_idx;
        }
    }

    void on_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (const auto *ev = begin; ev != end; ++ev) {
            const int64_t idx = static_cast<int64_t>(ev->t / slice_us);
            advance_to(idx);
            current_events.push_back(WireEventCD{
                static_cast<uint16_t>(ev->x),
                static_cast<uint16_t>(ev->y),
                static_cast<int16_t>(ev->p),
                static_cast<int64_t>(ev->t),
            });
        }
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
            write_frame(width, height, seq++, 0, start_us, start_us + slice_us, start_us + slice_us,
                        current_events, triggers);
        }
        pending_triggers.clear();
        current_events.clear();
        const std::vector<WireEventCD> empty_events;
        const std::vector<WireExtTrigger> empty_triggers;
        write_frame(width, height, seq++, kFlagFinal, 0, 0, 0, empty_events, empty_triggers);
    }
};

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
        if (arg == "--serial") {
            opt.serial = value(arg);
        } else if (arg == "--slice-us") {
            opt.slice_us = std::max(1, std::stoi(value(arg)));
        } else if (arg == "--wait-mode") {
            opt.wait_mode = value(arg);
        } else if (arg == "--poll-sleep-us") {
            opt.poll_sleep_us = std::max(0, std::stoi(value(arg)));
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
            opt.open_retries = std::max(1, std::stoi(value(arg)));
        } else if (arg == "--open-retry-delay-ms") {
            opt.open_retry_delay_ms = std::max(0, std::stoi(value(arg)));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.wait_mode != "blocking" && opt.wait_mode != "poll") {
        throw std::runtime_error("--wait-mode must be blocking or poll");
    }
    return opt;
}

void apply_optional_config(Metavision::Device &device, const Options &opt) {
    if (opt.enable_trigger_in) {
        auto *trigger = device.get_facility<Metavision::I_TriggerIn>();
        if (trigger) {
            const bool ok = trigger->enable(Metavision::I_TriggerIn::Channel::Main);
            std::cerr << "[bridge] trigger_in_main=" << (ok ? "ok" : "failed") << "\n";
        } else {
            std::cerr << "[bridge] trigger_in=missing\n";
        }
    }

    if (opt.enable_erc) {
        auto *erc = device.get_facility<Metavision::I_ErcModule>();
        if (erc) {
            const bool en = erc->enable(true);
            const bool sr = erc->set_cd_event_rate(opt.erc_rate);
            std::cerr << "[bridge] erc_enable=" << (en ? "ok" : "failed")
                      << " erc_rate=" << (sr ? "ok" : "failed") << "\n";
        } else {
            std::cerr << "[bridge] erc=missing\n";
        }
    }

    if (opt.enable_trail) {
        auto *trail = device.get_facility<Metavision::I_EventTrailFilterModule>();
        if (trail) {
            const auto type = parse_trail_type(opt.trail_type);
            const bool ty = trail->set_type(type);
            const bool th = trail->set_threshold(opt.trail_th_us);
            const bool en = trail->enable(true);
            std::cerr << "[bridge] trail_type=" << opt.trail_type << ":" << (ty ? "ok" : "failed")
                      << " trail_th=" << (th ? "ok" : "failed")
                      << " trail_enable=" << (en ? "ok" : "failed") << "\n";
        } else {
            std::cerr << "[bridge] trail=missing\n";
        }
    }

    if (!opt.biases.empty()) {
        auto *biases = device.get_facility<Metavision::I_LL_Biases>();
        if (biases) {
            for (const auto &kv : opt.biases) {
                try {
                    const bool ok = biases->set(kv.first, kv.second);
                    std::cerr << "[bridge] bias_" << kv.first << "=" << kv.second << ":"
                              << (ok ? "ok" : "failed") << "\n";
                } catch (const std::exception &e) {
                    std::cerr << "[bridge] bias_" << kv.first << "=" << kv.second << ":err:" << e.what() << "\n";
                }
            }
        } else {
            std::cerr << "[bridge] biases=missing\n";
        }
    }
}

std::unique_ptr<Metavision::Device> open_device(const Options &opt) {
    auto candidates = serial_candidates(opt.serial);
    for (int attempt = 1; attempt <= opt.open_retries; ++attempt) {
        for (const auto &candidate : candidates) {
            try {
                Metavision::DeviceConfig cfg;
                cfg.enable_biases_range_check_bypass(opt.bypass_bias_range);
                auto device = Metavision::DeviceDiscovery::open(candidate, cfg);
                if (device) {
                    std::cerr << "[bridge] opened serial=" << (candidate.empty() ? "first_available" : candidate)
                              << " attempt=" << attempt << "\n";
                    return device;
                }
            } catch (const Metavision::BaseException &e) {
                std::cerr << "[bridge] open attempt=" << attempt << " candidate=" << candidate
                          << " base_exception=" << e.what() << "\n";
            } catch (const std::exception &e) {
                std::cerr << "[bridge] open attempt=" << attempt << " candidate=" << candidate
                          << " exception=" << e.what() << "\n";
            }
        }
        if (attempt < opt.open_retries && opt.open_retry_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(opt.open_retry_delay_ms));
        }
    }
    return nullptr;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options opt = parse_args(argc, argv);
        auto device = open_device(opt);
        if (!device) {
            std::cerr << "[bridge] failed to open device serial=" << opt.serial << "\n";
            return 2;
        }

        apply_optional_config(*device, opt);

        auto *stream = device->get_facility<Metavision::I_EventsStream>();
        auto *decoder = device->get_facility<Metavision::I_EventsStreamDecoder>();
        auto *cd = device->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();
        auto *trigger_decoder = device->get_facility<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
        auto *geo = device->get_facility<Metavision::I_Geometry>();
        if (!stream || !decoder || !cd || !geo) {
            std::cerr << "[bridge] missing facilities stream=" << (stream ? "ok" : "missing")
                      << " decoder=" << (decoder ? "ok" : "missing")
                      << " cd=" << (cd ? "ok" : "missing")
                      << " geometry=" << (geo ? "ok" : "missing") << "\n";
            return 3;
        }

        SliceEmitter emitter;
        emitter.width = static_cast<uint32_t>(geo->get_width());
        emitter.height = static_cast<uint32_t>(geo->get_height());
        emitter.slice_us = opt.slice_us;
        emitter.emit_hello();

        cd->add_event_buffer_callback([&emitter](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            emitter.on_events(begin, end);
        });
        if (trigger_decoder) {
            trigger_decoder->add_event_buffer_callback(
                [&emitter](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                    emitter.on_triggers(begin, end);
                });
        } else {
            std::cerr << "[bridge] ext_trigger_decoder=missing\n";
        }

        std::cerr << "[bridge] start serial=" << opt.serial << " size=" << emitter.width << "x" << emitter.height
                  << " slice_us=" << opt.slice_us << " wait_mode=" << opt.wait_mode << "\n";

        stream->start();
        while (!g_stop.load(std::memory_order_relaxed)) {
            short ret = (opt.wait_mode == "blocking") ? stream->wait_next_buffer() : stream->poll_buffer();
            if (ret < 0) {
                break;
            }
            if (ret == 0) {
                if (opt.poll_sleep_us > 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(opt.poll_sleep_us));
                }
                continue;
            }
            auto raw_data = stream->get_latest_raw_data();
            if (raw_data) {
                decoder->decode(raw_data->data(), raw_data->data() + raw_data->size());
            }
        }
        try {
            stream->stop();
        } catch (...) {
        }
        emitter.finalize();
        std::cerr << "[bridge] stop\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[bridge] ERROR: " << e.what() << "\n";
        return 1;
    }
}

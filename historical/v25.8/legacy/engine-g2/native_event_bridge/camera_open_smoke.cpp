#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/driver/camera.h>

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

struct Options {
    std::string serial;
    int duration_s = 10;
};

Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--serial") {
            opt.serial = value("--serial");
        } else if (arg == "--duration-sec") {
            opt.duration_s = std::max(1, std::stoi(value("--duration-sec")));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: camera_open_smoke [--serial SERIAL] [--duration-sec N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opt;
}

} // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    try {
        const Options opt = parse_args(argc, argv);

        Metavision::Camera cam;
        if (opt.serial.empty()) {
            std::cout << "opening mode=first_available\n";
            cam = Metavision::Camera::from_first_available();
        } else {
            std::cout << "opening mode=serial serial=" << opt.serial << "\n";
            cam = Metavision::Camera::from_serial(opt.serial);
        }

        std::atomic<uint64_t> callbacks{0};
        std::atomic<uint64_t> events{0};
        std::atomic<int64_t> first_ts{-1};
        std::atomic<int64_t> last_ts{-1};

        cam.add_runtime_error_callback([](const Metavision::CameraException &e) {
            std::cerr << "runtime_error " << e.what() << "\n";
            g_stop.store(true, std::memory_order_relaxed);
        });

        cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            const auto n = static_cast<uint64_t>(end - begin);
            callbacks.fetch_add(1, std::memory_order_relaxed);
            events.fetch_add(n, std::memory_order_relaxed);
            if (n > 0) {
                int64_t expected = -1;
                first_ts.compare_exchange_strong(expected, begin->t);
                last_ts.store((end - 1)->t, std::memory_order_relaxed);
            }
        });

        std::cout << "geometry width=" << cam.geometry().width() << " height=" << cam.geometry().height() << "\n";
        const auto started = std::chrono::steady_clock::now();
        cam.start();
        std::cout << "started duration_s=" << opt.duration_s << "\n";

        uint64_t last_events = 0;
        uint64_t last_callbacks = 0;
        int tick = 0;
        while (!g_stop.load(std::memory_order_relaxed) && cam.is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            ++tick;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - started).count();
            const uint64_t e = events.load(std::memory_order_relaxed);
            const uint64_t c = callbacks.load(std::memory_order_relaxed);
            std::cout << "tick=" << tick
                      << " elapsed_s=" << elapsed
                      << " callbacks=" << c
                      << " events=" << e
                      << " delta_callbacks=" << (c - last_callbacks)
                      << " delta_events=" << (e - last_events)
                      << " first_ts=" << first_ts.load(std::memory_order_relaxed)
                      << " last_ts=" << last_ts.load(std::memory_order_relaxed)
                      << "\n";
            last_events = e;
            last_callbacks = c;
            if (elapsed >= opt.duration_s) {
                break;
            }
        }

        cam.stop();
        const auto ended = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(ended - started).count();
        std::cout << "final ok=1 elapsed_s=" << elapsed
                  << " callbacks=" << callbacks.load(std::memory_order_relaxed)
                  << " events=" << events.load(std::memory_order_relaxed)
                  << " first_ts=" << first_ts.load(std::memory_order_relaxed)
                  << " last_ts=" << last_ts.load(std::memory_order_relaxed)
                  << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "final ok=0 error=" << e.what() << "\n";
        return 2;
    }
}

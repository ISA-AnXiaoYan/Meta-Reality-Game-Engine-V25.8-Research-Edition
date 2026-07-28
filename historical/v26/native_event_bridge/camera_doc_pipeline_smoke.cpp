#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/flip_x_algorithm.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

struct Options {
    std::string serial;
    int duration_s = 5;
    bool display = false;
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
        } else if (arg == "--display") {
            opt.display = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: camera_doc_pipeline_smoke [--serial SERIAL] [--duration-sec N] [--display]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return opt;
}

class EventAnalyzer {
public:
    void analyze_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        const auto n = static_cast<uint64_t>(end - begin);
        callbacks_.fetch_add(1, std::memory_order_relaxed);
        events_.fetch_add(n, std::memory_order_relaxed);
        if (n > 0) {
            int64_t expected = -1;
            first_ts_.compare_exchange_strong(expected, begin->t);
            last_ts_.store((end - 1)->t, std::memory_order_relaxed);
        }
    }

    uint64_t callbacks() const {
        return callbacks_.load(std::memory_order_relaxed);
    }

    uint64_t events() const {
        return events_.load(std::memory_order_relaxed);
    }

    int64_t first_ts() const {
        return first_ts_.load(std::memory_order_relaxed);
    }

    int64_t last_ts() const {
        return last_ts_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> callbacks_{0};
    std::atomic<uint64_t> events_{0};
    std::atomic<int64_t> first_ts_{-1};
    std::atomic<int64_t> last_ts_{-1};
};

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

        cam.add_runtime_error_callback([](const Metavision::CameraException &e) {
            std::cerr << "runtime_error " << e.what() << "\n";
            g_stop.store(true, std::memory_order_relaxed);
        });

        EventAnalyzer analyzer;
        cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            analyzer.analyze_events(begin, end);
        });

        const int width = cam.geometry().width();
        const int height = cam.geometry().height();
        std::cout << "geometry width=" << width << " height=" << height << "\n";

        const std::uint32_t accumulation_us = 20000;
        const double fps = 50.0;
        Metavision::PeriodicFrameGenerationAlgorithm frame_gen(width, height, accumulation_us, fps);
        Metavision::FlipXAlgorithm flip_x(width - 1);
        std::atomic<uint64_t> algorithm_callbacks{0};
        std::atomic<uint64_t> frames{0};

        std::unique_ptr<Metavision::Window> window;
        if (opt.display) {
            window = std::make_unique<Metavision::Window>(
                "Metavision 4.6.2 doc pipeline smoke", width, height, Metavision::BaseWindow::RenderMode::BGR);
            window->set_keyboard_callback([&](Metavision::UIKeyEvent key, int, Metavision::UIAction action, int) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    g_stop.store(true, std::memory_order_relaxed);
                }
            });
        }

        frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
            frames.fetch_add(1, std::memory_order_relaxed);
            if (window) {
                window->show(frame);
            }
        });

        std::vector<Metavision::EventCD> flipped;
        flipped.reserve(8192);
        cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            flipped.clear();
            flip_x.process_events(begin, end, std::back_inserter(flipped));
            frame_gen.process_events(flipped.begin(), flipped.end());
            algorithm_callbacks.fetch_add(1, std::memory_order_relaxed);
        });

        const auto started = std::chrono::steady_clock::now();
        cam.start();
        std::cout << "started duration_s=" << opt.duration_s
                  << " display=" << (opt.display ? 1 : 0)
                  << " algorithm=flip_x frame_accumulation_us=" << accumulation_us
                  << " frame_fps=" << fps << "\n";

        int tick = 0;
        uint64_t last_events = 0;
        uint64_t last_frames = 0;
        while (!g_stop.load(std::memory_order_relaxed) && cam.is_running()) {
            if (window) {
                Metavision::EventLoop::poll_and_dispatch(10);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - started).count();
            if (elapsed >= tick + 1) {
                ++tick;
                const uint64_t e = analyzer.events();
                const uint64_t f = frames.load(std::memory_order_relaxed);
                std::cout << "tick=" << tick
                          << " elapsed_s=" << elapsed
                          << " callbacks=" << analyzer.callbacks()
                          << " events=" << e
                          << " delta_events=" << (e - last_events)
                          << " algorithm_callbacks=" << algorithm_callbacks.load(std::memory_order_relaxed)
                          << " frames=" << f
                          << " delta_frames=" << (f - last_frames)
                          << " first_ts=" << analyzer.first_ts()
                          << " last_ts=" << analyzer.last_ts()
                          << "\n";
                last_events = e;
                last_frames = f;
            }
            if (elapsed >= opt.duration_s) {
                break;
            }
        }

        cam.stop();
        std::cout << "final ok=1 callbacks=" << analyzer.callbacks()
                  << " events=" << analyzer.events()
                  << " algorithm_callbacks=" << algorithm_callbacks.load(std::memory_order_relaxed)
                  << " frames=" << frames.load(std::memory_order_relaxed)
                  << " first_ts=" << analyzer.first_ts()
                  << " last_ts=" << analyzer.last_ts()
                  << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "final ok=0 error=" << e.what() << "\n";
        return 2;
    }
}

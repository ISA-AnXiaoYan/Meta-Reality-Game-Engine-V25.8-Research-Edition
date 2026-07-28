cmake_minimum_required(VERSION 3.10)
project(hal_event_pipe_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(MetavisionHAL REQUIRED)
find_package(MetavisionSDK COMPONENTS driver core ui REQUIRED)
find_package(Threads REQUIRED)

add_executable(hal_event_pipe_bridge hal_event_pipe_bridge.cpp)
target_link_libraries(hal_event_pipe_bridge PRIVATE Metavision::HAL_discovery Threads::Threads)

add_executable(hal_event_fifo_broker hal_event_fifo_broker.cpp)
target_link_libraries(hal_event_fifo_broker PRIVATE Metavision::HAL_discovery Threads::Threads)

add_executable(event_file_fifo_broker event_file_fifo_broker.cpp)
target_link_libraries(event_file_fifo_broker PRIVATE MetavisionSDK::driver MetavisionSDK::core Threads::Threads)

add_executable(camera_open_smoke camera_open_smoke.cpp)
target_link_libraries(camera_open_smoke PRIVATE MetavisionSDK::driver MetavisionSDK::core Threads::Threads)

add_executable(camera_doc_pipeline_smoke camera_doc_pipeline_smoke.cpp)
target_link_libraries(camera_doc_pipeline_smoke PRIVATE MetavisionSDK::driver MetavisionSDK::core MetavisionSDK::ui Threads::Threads)

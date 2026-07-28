# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track simple, non colliding objects.
You can use it, for example, with the reference file sparklers.raw.
"""

import cv2
import numpy as np
import csv

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import SpatterTrackerAlgorithm, SpatterTrackingConfig, ClusterTrajectories, \
    draw_tracking_results, EventSpatterClusterBuffer
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

ev_filter_type_dict = {"NoFilter": SpatterTrackingConfig.FilterType.NoFilter,
                       "FilterNegative": SpatterTrackingConfig.FilterType.FilterNegative,
                       "FilterPositive": SpatterTrackingConfig.FilterType.FilterPositive,
                       "SeparateFilter": SpatterTrackingConfig.FilterType.SeparateFilter}


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Spatter Tracking sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
             "If it's a camera serial number, it will try to open that camera instead.")
    base_options.add_argument('-s', '--process-from', dest='process_from', type=int, default=0,
                              help='Start time to process events (in us).')
    base_options.add_argument('-e', '--process-to', dest='process_to', type=int, default=None,
                              help='End time to process events (in us).')
    base_options.add_argument('--stc-threshold', dest='stc_threshold', type=int, default=0,
                              help='STC: filtering threshold delay (in us). 0 means no filtering will be applied.')

    # Algorithm options
    algo_options = parser.add_argument_group('Algorithm options')
    algo_options.add_argument(
        '--polarity-filter', dest='filtered_polarity_str', type=str,
        default="FilterNegative",
        choices=["NoFilter", "FilterNegative", "FilterPositive", "SeparateFilter"],
        help='Polarity of events to filter out.')
    algo_options.add_argument('--cell-width', dest='cell_width', type=int,
                              default=7, help='Cell width used for clustering (in pixels).')
    algo_options.add_argument('--cell-height', dest='cell_height', type=int,
                              default=7, help='Cell height used for clustering (in pixels).')
    algo_options.add_argument('--processing-accumulation-time', dest='accumulation_time_us', type=int,
                              default=5000, help='Processing accumulation time (in us).')
    algo_options.add_argument('--static-memory', dest='static_memory_us', type=int,
                              default=1000000, help='Memory of static clusters (us).')
    algo_options.add_argument('--untracked-threshold', dest='untracked_ths', type=int, default=5,
                              help='Maximum number of times a cluster can stay untracked before being removed.')
    algo_options.add_argument('--min-track-time', dest='min_track_time', type=int, default=1000,
                              help='Minimum time for a detected cluster to be tracked.')
    algo_options.add_argument('-a', '--activation-threshold', dest='activation_ths', type=int, default=10,
                              help='Minimum number of events in a cell to consider it as active.')
    algo_options.add_argument(
        '--apply-filter', dest='apply_filter', default=False, action='store_true',
        help='If set, then the cell activation threshold considers only one event per pixel.')
    algo_options.add_argument('-D', '--max-distance', dest='max_distance', type=int,
                              default=50, help='Maximum distance for clusters association (in pixels).')
    algo_options.add_argument(
        '--max-size-variation', dest='max_size_variation', type=int, default=100,
        help="Maximum size (in pixels) to allow matching two clusters.")
    algo_options.add_argument(
        '--min-dist-moving-obj', dest='min_dist_moving_obj', type=int, default=3,
        help="Minimum distance (in pixels) to consider a cluster moving (else, it is considered a static object which we don't want to track).")
    algo_options.add_argument('--min-size', dest='min_size', type=int,
                              default=10, help='Minimal size of an object to track (in pixels).')
    algo_options.add_argument('--max-size', dest='max_size', type=int,
                              default=300, help='Maximal size of an object to track (in pixels).')
    algo_options.add_argument('--nozone', nargs='+', action='append', type=int,
                              metavar='center_x center_y radius inside', default=[],
                              help='Add a no-zone ROI with center coordinates, radius '
                                   'and inside(1)/outside(0) arguments')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to the resulting video. A frame is generated every time the tracking callback is called.")
    outcome_options.add_argument(
        '-l', '--log-results', dest='out_log', type=str, default="",
        help="File to save the output of tracking.")
    outcome_options.add_argument(
        '--traj-duration', dest='traj_duration', type=int, default=0,
        help="Last X us of trajectory to draw for each tracked cluster. 0 means no trajectory will be drawn.")

    # Replay Option
    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument(
        '-f', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")

    args = parser.parse_args()

    if args.process_to and args.process_from > args.process_to:
        print(f"The processing time interval is not valid. [{args.process_from,}, {args.process_to}]")
        exit(1)

    return args


def main():
    """ Main """
    args = parse_args()

    # [SLICER_INIT_BEGIN]
    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(input_path=args.event_file_path, start_ts=args.process_from,
                                 max_duration=args.process_to - args.process_from if args.process_to else None,
                                 delta_t=args.accumulation_time_us)

    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    # [SLICER_INIT_END]

    height, width = mv_iterator.get_size()  # Camera Geometry
    print("Camera dimensions : ", height, "x", width)

    # Spatter Tracking Algorithm
    spatter_tracker = SpatterTrackerAlgorithm(width=width, height=height,
                                              cell_width=args.cell_width,
                                              cell_height=args.cell_height,
                                              untracked_threshold=args.untracked_ths,
                                              activation_threshold=args.activation_ths,
                                              apply_filter=args.apply_filter,
                                              max_distance=args.max_distance,
                                              min_size=args.min_size,
                                              max_size=args.max_size,
                                              min_track_time=args.min_track_time,
                                              static_memory_us=args.static_memory_us,
                                              max_size_variation=args.max_size_variation,
                                              min_dist_moving_obj_pxl=args.min_dist_moving_obj,
                                              filter_type=ev_filter_type_dict[args.filtered_polarity_str]
                                              )

    # Output buffer
    clusters = EventSpatterClusterBuffer()

    do_filtering = args.stc_threshold > 0

    if do_filtering:
        stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_threshold, False)

    # Process each defined no-zone ROI
    input_nozone_rois = args.nozone
    for roi in input_nozone_rois:
        center_x, center_y, radius, inside = roi
        spatter_tracker.add_nozone(np.array([center_x, center_y]), radius, inside)
        print("The region center=(", center_x, ",", center_y, ") radius=",
              radius, ", inside:", inside, "isn't processed")

    print(
        "Red circles define regions where clusters found inside are filtered, while green ones filter clusters outside the region.")

    # Event Frame Generator
    events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, args.accumulation_time_us)
    output_img = np.zeros((height, width, 3), np.uint8)

    # Window - Graphical User Interface (Display spatter tracking results and process keyboard events)
    with MTWindow(title="Spatter Tracking", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        window.show_async(output_img)

        if args.out_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = args.out_video + ".avi"
            video_writer = cv2.VideoWriter(video_name, fourcc, 20, (width, height))

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        log = []
        tracked_clusters = ClusterTrajectories(args.traj_duration)

        def draw_no_track_zones(img):
            for roi in input_nozone_rois:
                center_x, center_y, radius, inside = roi
                cv2.circle(img, (center_x, center_y), radius, (0, 0, 255) if inside else (0, 255, 0))

        # [TRACKING_MAIN_LOOP_BEGIN]
        # Process events
        filtered_evs = SpatioTemporalContrastAlgorithm.get_empty_output_buffer() if do_filtering else None

        for evs in mv_iterator:
            ts = mv_iterator.get_current_time()

            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Process events
            if do_filtering:
                stc_filter.process_events(evs, filtered_evs)

            buffer = filtered_evs if do_filtering else evs

            events_frame_gen_algo.process_events(buffer)
            spatter_tracker.process_events(buffer, ts, clusters)

            clusters_np = clusters.numpy()
            for cluster in clusters_np:
                log.append([ts, cluster['id'], int(cluster['x']), int(cluster['y']), int(cluster['width']),
                            int(cluster['height'])])
            events_frame_gen_algo.generate(ts, output_img)
            tracked_clusters.update_trajectories(ts, clusters)
            draw_tracking_results(ts, clusters, output_img)
            draw_no_track_zones(output_img)
            tracked_clusters.draw(output_img)

            window.show_async(output_img)
            if args.out_video:
                video_writer.write(output_img)

            if window.should_close():
                break
        # [TRACKING_MAIN_LOOP_END]

        print("Number of tracked clusters: {}".format(spatter_tracker.get_cluster_count))

        if args.out_video:
            video_writer.release()
            print("Video has been saved in " + video_name)

        if args.out_log:
            with open(args.out_log, mode='w') as f:
                data_writer = csv.writer(f)
                data_writer.writerow(['timestamp', 'id', 'x', 'y', 'width', 'height'])
                data_writer.writerows(log)


if __name__ == "__main__":
    main()

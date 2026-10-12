"""Stable control-log and live-status schemas, separate from control execution.

These builders run after command application. They do not authorize motion.
All cycle state is supplied explicitly, preserving historical column semantics.
"""
import time
from .corner_runtime import STATIONARY_LOG_FIELDS

ENCODER_LOG_FIELDS = (
    'encoder_1', 'encoder_2', 'encoder_3', 'encoder_4',
    'encoder_received_monotonic', 'encoder_age_s', 'encoder_valid', 'encoder_stale',
)

APPLICATION_LOG_FIELDS = (
    'stationary_application_monotonic', 'stationary_application_reason',
    'semantic_application_age_s', 'frame_application_age_s', 'imu_application_age_s',
)

def control_log_fields():
    return [
            *STATIONARY_LOG_FIELDS,
            *ENCODER_LOG_FIELDS,
            *APPLICATION_LOG_FIELDS,
            "vehicle_profile",
            "vehicle_backend",
            "sample",
            "timestamp_s",
            "frame_age_s",
            "capture_interval_s",
            "loop_start_interval_s",
            "source_wait_ms",
            "analysis_ms",
            "control_render_ms",
            "control_gate_ms",
            "hardware_apply_ms",
            "birdeye_render_ms",
            "inference_ms",
            "semantic_inference_ms",
            "semantic_result_age_s",
            "semantic_result_age_exact_s",
            "semantic_track_colors",
            "semantic_allowed_white_pixels",
            "semantic_yellow_pixels",
            "semantic_sequence",
            "semantic_preview_x",
            "semantic_preview_y",
            "semantic_stability_mode",
            "semantic_filter_changed_ratio",
            "semantic_fusion_source",
            "semantic_precision",
            "semantic_requested_precision",
            "semantic_overlap_ratio",
            "semantic_drivable_ratio",
            "boundary_ms",
            "boundary_source",
            "boundary_confidence",
            "boundary_left_rows",
            "boundary_right_rows",
            "lane_width_ratio",
            "boundary_visible_ratio",
            "ego_yellow_ratio",
            "yellow_hazard",
            "valid",
            "confidence",
            "estimate_reason",
            "lateral_error",
            "heading_error",
            "near_heading_error",
            "action",
            "steering",
            "tight_turn_factor",
            "left_speed",
            "right_speed",
            "left_pwm",
            "right_pwm",
            "front_left_pwm",
            "rear_left_pwm",
            "front_right_pwm",
            "rear_right_pwm",
            "reason",
            "corner_continuation_active",
            "corner_continuation_holding",
            "corner_continuation_hold_age_s",
            "corner_continuation_progress_age_s",
            "corner_continuation_best_heading",
            "corner_continuation_best_lateral",
            "corner_apex_active",
            "corner_apex_age_s",
            "corner_apex_trigger_reason",
            "corner_apex_completion_reason",
            "corner_apex_exit_valid_count",
            "corner_apex_both_valid_count",
            "motion_gate_ready",
            "motion_gate_valid_frames",
            "cloud_phase",
            "cloud_reason",
            "cloud_event_id",
            "archive_queue_depth",
            "archive_dropped_frames",
            "log_queue_depth",
            "log_dropped_rows",
            "live_dropped_updates",
            "debug_dropped_frames",
        ]


def build_control_row(*,
    stationary_state,
    analysis_time,
    application_now,
    application_reason,
    archive_state,
    birdeye_render_time,
    boundary_result,
    boundary_time,
    cloud_coordinator,
    command,
    control_render_time,
    corner_state,
    debug_state,
    detector,
    dt,
    encoder_sample,
    encoder_ticks,
    estimate,
    frame_age,
    frame_application_age,
    gate_time,
    hardware_apply_time,
    imu_application_age,
    inference_time,
    live_state,
    log_state,
    loop_start_interval,
    motion_state,
    sample,
    semantic_application_age,
    source_wait_time,
    timestamp,
    vehicle_summary,
    wheel_state,
):
    return {
                **{field: stationary_state.get(field) for field in STATIONARY_LOG_FIELDS},
                'stationary_application_monotonic': application_now,
                'stationary_application_reason': application_reason,
                'semantic_application_age_s': semantic_application_age,
                'frame_application_age_s': frame_application_age,
                'imu_application_age_s': imu_application_age,
                **{'encoder_'+str(index+1): value for index, value in enumerate(encoder_ticks)},
                'encoder_received_monotonic': encoder_sample.get('received_monotonic'),
                'encoder_age_s': encoder_sample.get('age_s'),
                'encoder_valid': encoder_sample.get('valid') is True,
                'encoder_stale': encoder_sample.get('stale') is not False,
                "semantic_sequence": getattr(boundary_result, 'semantic_sequence', None),
                "semantic_preview_x": (boundary_result.semantic_preview_point[0]
                    if boundary_result is not None and boundary_result.semantic_preview_point is not None else None),
                "semantic_preview_y": (boundary_result.semantic_preview_point[1]
                    if boundary_result is not None and boundary_result.semantic_preview_point is not None else None),
                "semantic_result_age_exact_s": getattr(boundary_result, 'semantic_result_age_seconds', None),
                "semantic_track_colors": getattr(boundary_result,'semantic_track_colors',False),
                "semantic_allowed_white_pixels": getattr(boundary_result,'semantic_allowed_white_pixels',0),
                "semantic_yellow_pixels": getattr(boundary_result,'semantic_yellow_pixels',0),
                "semantic_stability_mode": (detector.stabilizer.state['mode']
                    if hasattr(detector, 'stabilizer') else None),
                "semantic_filter_changed_ratio": (detector.stabilizer.state['changed_ratio']
                    if hasattr(detector, 'stabilizer') else None),
                "vehicle_profile": vehicle_summary["profile"],
                "vehicle_backend": vehicle_summary["backend"],
                "sample": sample,
                "timestamp_s": round(float(timestamp), 4),
                "frame_age_s": (
                    None if frame_age is None else round(float(frame_age), 4)
                ),
                "capture_interval_s": (
                    None if dt is None else round(float(dt), 4)
                ),
                "loop_start_interval_s": (
                    None
                    if loop_start_interval is None
                    else round(float(loop_start_interval), 4)
                ),
                "source_wait_ms": round(source_wait_time * 1000, 3),
                "analysis_ms": round(analysis_time * 1000, 3),
                "control_render_ms": round(control_render_time * 1000, 3),
                "control_gate_ms": round(gate_time * 1000, 3),
                "hardware_apply_ms": round(hardware_apply_time * 1000, 3),
                "birdeye_render_ms": round(
                    birdeye_render_time * 1000, 3
                ),
                "inference_ms": round(inference_time * 1000, 3),
                "semantic_inference_ms": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_inference_seconds is None
                    else round(
                        boundary_result.semantic_inference_seconds * 1000,
                        3,
                    )
                ),
                "semantic_result_age_s": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_result_age_seconds is None
                    else round(
                        boundary_result.semantic_result_age_seconds,
                        4,
                    )
                ),
                "semantic_fusion_source": (
                    None
                    if boundary_result is None
                    else boundary_result.semantic_fusion_source
                ),
                "semantic_precision": (
                    None
                    if boundary_result is None
                    else boundary_result.semantic_precision
                ),
                "semantic_requested_precision": (
                    None
                    if boundary_result is None
                    else boundary_result.semantic_requested_precision
                ),
                "semantic_overlap_ratio": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_overlap_ratio is None
                    else round(boundary_result.semantic_overlap_ratio, 5)
                ),
                "semantic_drivable_ratio": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_drivable_ratio is None
                    else round(boundary_result.semantic_drivable_ratio, 5)
                ),
                "boundary_ms": round(boundary_time * 1000, 3),
                "boundary_source": (
                    None if boundary_result is None else boundary_result.source
                ),
                "boundary_confidence": (
                    None
                    if boundary_result is None
                    else round(boundary_result.confidence, 5)
                ),
                "boundary_left_rows": (
                    None
                    if boundary_result is None
                    else boundary_result.observed_left_rows
                ),
                "boundary_right_rows": (
                    None
                    if boundary_result is None
                    else boundary_result.observed_right_rows
                ),
                "lane_width_ratio": (
                    None
                    if boundary_result is None
                    else round(boundary_result.lane_width_ratio, 5)
                ),
                "boundary_visible_ratio": (
                    None
                    if boundary_result is None
                    else round(boundary_result.boundary_visible_ratio, 5)
                ),
                "ego_yellow_ratio": (
                    None
                    if boundary_result is None
                    else round(boundary_result.ego_yellow_ratio, 5)
                ),
                "yellow_hazard": (
                    None
                    if boundary_result is None
                    else boundary_result.yellow_hazard
                ),
                "valid": estimate.valid,
                "confidence": round(estimate.confidence, 5),
                "estimate_reason": estimate.reason,
                "lateral_error": round(estimate.lateral_error, 5),
                "heading_error": round(estimate.heading_error, 5),
                "near_heading_error": round(
                    estimate.near_heading_error, 5
                ),
                "action": command.action,
                "steering": round(command.steering, 5),
                "tight_turn_factor": (
                    None
                    if command.tight_turn_factor is None
                    else round(command.tight_turn_factor, 5)
                ),
                "left_speed": round(command.left_speed, 5),
                "right_speed": round(command.right_speed, 5),
                "left_pwm": wheel_state["left_pwm"],
                "right_pwm": wheel_state["right_pwm"],
                "front_left_pwm": wheel_state["front_left_pwm"],
                "rear_left_pwm": wheel_state["rear_left_pwm"],
                "front_right_pwm": wheel_state["front_right_pwm"],
                "rear_right_pwm": wheel_state["rear_right_pwm"],
                "reason": command.reason,
                "corner_continuation_active": corner_state["active"],
                "corner_continuation_holding": corner_state["holding"],
                "corner_continuation_hold_age_s": (
                    None
                    if corner_state["hold_age_seconds"] is None
                    else round(corner_state["hold_age_seconds"], 5)
                ),
                "corner_continuation_progress_age_s": (
                    None
                    if corner_state["progress_age_seconds"] is None
                    else round(corner_state["progress_age_seconds"], 5)
                ),
                "corner_continuation_best_heading": (
                    None
                    if corner_state["best_heading_magnitude"] is None
                    else round(corner_state["best_heading_magnitude"], 5)
                ),
                "corner_continuation_best_lateral": (
                    None
                    if corner_state["best_lateral_magnitude"] is None
                    else round(corner_state["best_lateral_magnitude"], 5)
                ),
                "corner_apex_active": corner_state["apex_active"],
                "corner_apex_age_s": (
                    None
                    if corner_state["apex_age_seconds"] is None
                    else round(corner_state["apex_age_seconds"], 5)
                ),
                "corner_apex_trigger_reason": corner_state[
                    "apex_trigger_reason"
                ],
                "corner_apex_completion_reason": corner_state[
                    "apex_completion_reason"
                ],
                "corner_apex_exit_valid_count": corner_state[
                    "apex_exit_valid_count"
                ],
                "corner_apex_both_valid_count": corner_state[
                    "apex_both_valid_count"
                ],
                "motion_gate_ready": motion_state["ready"],
                "motion_gate_valid_frames": motion_state["consecutive_valid"],
                "cloud_phase": cloud_coordinator.get_state()["phase"],
                "cloud_reason": cloud_coordinator.get_state()["reason"],
                "cloud_event_id": cloud_coordinator.get_state()["event_id"],
                "archive_queue_depth": archive_state["queue_depth"],
                "archive_dropped_frames": archive_state["dropped_frames"],
                "log_queue_depth": log_state["queue_depth"],
                "log_dropped_rows": log_state["dropped_rows"],
                "live_dropped_updates": live_state["dropped_updates"],
                "debug_dropped_frames": debug_state["dropped_frames"],
            }


def build_live_status(*,
    archive,
    args,
    calibration_path,
    camera_config,
    cloud_coordinator,
    config_path,
    corner_state,
    debug_writer,
    detector,
    driver,
    live_publisher,
    log_writer,
    motion_state,
    row,
    runtime_config,
    sensor_recording_state,
    sensor_writer,
    stationary_runtime,
    stationary_state,
    telemetry_publisher,
    vehicle_summary,
    watchdog,
    wheel_state,
):
    return {
                    "vehicle": vehicle_summary,
                    "mode": wheel_state["mode"],
                    "source": str(
                        args.video or f"camera:{camera_config.get('index', 0)}"
                    ),
                    "config": str(config_path),
                    "calibration": (
                        str(calibration_path) if calibration_path else None
                    ),
                    "last_result": row,
                    "wheel_driver": driver.get_state(),
                    "watchdog": watchdog.get_state(),
                    "corner_continuation": corner_state,
                    "motion_gate": motion_state,
                    "cloud_arbitration": cloud_coordinator.get_state(),
                    "stationary_corner": None if stationary_runtime is None else stationary_state,
                    'sensor_recording':dict(local=None if sensor_writer is None else sensor_writer.get_state(),
                        ros=sensor_recording_state,required=runtime_config.get('require_sensor_recording',False),
                        serial_publisher=None if telemetry_publisher is None else telemetry_publisher.get_state()),
                    "run_archive": (
                        None if not archive.enabled else str(archive.run_dir)
                    ),
                    "diagnostics": {
                        "archive": archive.get_state(),
                        "control_log": log_writer.get_state(),
                        "live_publisher": live_publisher.get_state(),
                        "debug_frames": debug_writer.get_state(),
                        "yolopv2_fusion": (
                            detector.get_state()
                            if hasattr(detector, "get_state")
                            else {"enabled": False}
                        ),
                    },
                }

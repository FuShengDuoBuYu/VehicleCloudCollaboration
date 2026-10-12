"""Assemble components with explicit chassis factory; default remains opt-in."""
from __future__ import annotations

from autodrive.control.watchdog import CommandWatchdog
from autodrive.control.wheel_driver import SafeWheelDriver
from autodrive.control.wheel_driver import WheelMappingConfig
from autodrive.control.field_trial import validate_field_trial_config
from autodrive.control.field_trial import validate_stationary_corner_config
from autodrive.control.lane_centering import LCCConfig
from autodrive.control.lane_centering import LaneCenteringController
from autodrive.control.lane_centering import RoadCenterlineEstimator
from autodrive.perception.outer_loop import OuterLoopBoundaryConfig
from autodrive.perception.outer_loop import OuterLoopBoundaryTracker
from autodrive.perception.perspective import PerspectiveMapper
from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig
from autodrive.perception.yolopv2_fusion import YOLOPv2FusionDetector
from autodrive.runtime.configuration import cloud_configuration
from autodrive.runtime.configuration import repo_path
from autodrive.runtime.perception_pipeline import SurfaceOnlyDetector
from vehicle_control.factory import create_chassis
from vehicle_control.factory import validate_motion_configuration


def build_components(config, motors_enabled, calibration_path, field_trial=False, *,
                     chassis_factory=create_chassis,
                     motion_validator=validate_motion_configuration):
    validate_stationary_corner_config(config, motors_enabled=motors_enabled)
    if field_trial:
        validate_field_trial_config(config)
    cloud_configuration(config)
    perception = config.get("perception", {})
    primary = perception.get("mode", "classical") == "yolopv2"
    if perception.get("mode", "classical") not in {"classical", "yolopv2"}:
        raise ValueError("perception.mode must be classical or yolopv2")
    if primary:
        if not perception.get("yolopv2", {}).get("enabled", False):
            raise ValueError("YOLOPv2 primary mode requires an enabled model")
        if config.get("outer_loop", {}).get("enabled", False):
            raise ValueError("YOLOPv2 primary mode requires outer_loop.enabled=false")
        if config.get("safety", {}).get("corner_continuation", {}).get("enabled", False):
            raise ValueError("YOLOPv2 primary mode requires corner_continuation.enabled=false")
    outer_loop_config = OuterLoopBoundaryConfig(**config.get("outer_loop", {}))
    if not primary and not outer_loop_config.enabled:
        raise ValueError("outer_loop must be enabled for onboard LCC")
    if (
        not primary and outer_loop_config.navigation_mode == "boundary"
        and outer_loop_config.include_lane_mask
    ):
        raise ValueError("boundary mode requires include_lane_mask=false")
    detector = chassis = driver = watchdog = None
    try:
        mask_width = int(perception.get("mask_width", 320))
        mask_height = int(perception.get("mask_height", 180))
        yolopv2_settings = dict(perception.get("yolopv2", {}))
        if bool(yolopv2_settings.get("enabled", False)):
            weights = repo_path(yolopv2_settings.get("weights"))
            yolopv2_settings["weights"] = "" if weights is None else str(weights)
            if bool(yolopv2_settings.get("adaptive_precision", False)):
                int8_weights = repo_path(yolopv2_settings.get("int8_weights"))
                yolopv2_settings["int8_weights"] = (
                    "" if int8_weights is None else str(int8_weights)
                )
            detector_class = YOLOPv2FusionDetector
            if primary:
                from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
                detector_class = YOLOPv2SemanticDetector
            detector_options = ({'outer_loop_config': perception.get('semantic_outer_loop'),
                                 'stability_config': perception.get('semantic_stability'),
                                 'track_color_config': perception.get('track_colors')}
                                if primary else {})
            if primary:
                from autodrive.perception.semantic_outer_route import route_options
                detector_options['outer_loop_config'] = route_options(config)
            if primary and perception.get('track_colors',{}).get('enabled') and calibration_path is not None:
                raise ValueError('track colors require original camera coordinates without perspective warp')
            detector = detector_class(
                YOLOPv2FusionConfig(**yolopv2_settings),
                output_width=mask_width,
                output_height=mask_height,
                **detector_options,
            )
        else:
            detector = SurfaceOnlyDetector(width=mask_width, height=mask_height)

        estimator = RoadCenterlineEstimator(**config.get("centerline", {}))
        controller = LaneCenteringController(LCCConfig(**config.get("lcc", {})))
        boundary_tracker = (
            OuterLoopBoundaryTracker(outer_loop_config)
            if outer_loop_config.enabled
            else None
        )
        mapper = (
            PerspectiveMapper.from_yaml(calibration_path)
            if calibration_path is not None
            else None
        )

        chassis = None
        if motors_enabled:
            if not field_trial:
                motion_validator(config.get("vehicle", {}), config.get("wheels", {}))
            chassis = chassis_factory(config.get("vehicle", {}), require_motion_calibrated=not field_trial)
        driver = SafeWheelDriver(
            chassis=chassis,
            motors_enabled=motors_enabled,
            config=WheelMappingConfig(**config.get("wheels", {})),
        )
        safety = config.get("safety", {})
        watchdog = CommandWatchdog(
            driver,
            timeout=float(safety.get("watchdog_timeout", 0.40)),
            check_interval=float(safety.get("watchdog_check_interval", 0.05)),
        )
        return detector, estimator, controller, mapper, boundary_tracker, driver, watchdog
    except BaseException as exc:
        cleanup_errors = []
        actions = []
        if watchdog is not None:
            actions.append(('watchdog.close', watchdog.close))
        if driver is not None:
            actions.append(('driver.stop', lambda: driver.stop('component initialization failed')))
        elif chassis is not None:
            actions.append(('chassis.stop', chassis.stop))
        if chassis is not None and hasattr(chassis, 'close'):
            actions.append(('chassis.close', lambda: chassis.close(stop=False)))
        if detector is not None and hasattr(detector, 'close'):
            actions.append(('detector.close', detector.close))
        for name, action in actions:
            try:
                action()
            except Exception as cleanup_exc:
                cleanup_errors.append(dict(action=name, type=type(cleanup_exc).__name__,
                                           message=str(cleanup_exc)))
        if cleanup_errors:
            exc.component_cleanup_errors = cleanup_errors
        raise


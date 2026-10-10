import math
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
from vehicle_control.base import FourWheelChassis, ServoGimbal
from vehicle_control.factory import create_chassis, create_gimbal, validate_motion_configuration, emergency_stop
from vehicle_control.platforms.rosmaster import RosmasterChassis, RosmasterGimbal


class RecordingController:
    def __init__(self):
        self.commands = []
        self.closed = False
        self.failures = 0
    def set_motor(self, *values):
        self.commands.append(values)
        if self.failures:
            self.failures -= 1
            raise OSError("serial disconnected")
    def set_pwm_servo(self, channel, angle):
        self.commands.append((channel, angle))
    def close(self):
        self.closed = True


class HardwareSafetyTests(unittest.TestCase):
    def profile(self, **status):
        return {"backend": "rosmaster", "status": status,
                "chassis": {"command_limit": 100}}

    def test_calibration_values_must_be_actual_booleans(self):
        for value in ("false", "true", 1, 0, [], None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_motion_configuration(self.profile(motion_calibrated=value), {"pwm_limit": 20})
                with self.assertRaises(ValueError):
                    create_gimbal(self.profile(gimbal_calibrated=value), controller=RecordingController())

    def test_invalid_chassis_configuration_never_opens_serial(self):
        for options in ({"motor_order": (0,0,2,3)}, {"motor_order": (0,1,2,3.5)},
                        {"command_limit": 101}, {"command_limit": 0},
                        {"command_limit": 20.5}, {"wheel_signs": (1,1,1,1.5)}):
            with self.subTest(options=options), patch("vehicle_control.platforms.rosmaster._new_rosmaster") as connect:
                with self.assertRaises(ValueError):
                    RosmasterChassis(**options)
                connect.assert_not_called()

    def test_invalid_gimbal_configuration_never_opens_serial(self):
        for options in ({"pan_servo_id": 5}, {"tilt_servo_id": 1},
                        {"pan_limits": (180,0)}, {"tilt_limits": (0,100.5)}):
            with self.subTest(options=options), patch("vehicle_control.platforms.rosmaster._new_rosmaster") as connect:
                with self.assertRaises(ValueError):
                    RosmasterGimbal(**options)
                connect.assert_not_called()

    def test_invalid_pose_does_not_partially_write_pan(self):
        controller=RecordingController()
        gimbal=RosmasterGimbal(controller=controller, tilt_limits=(0,100))
        with self.assertRaises(ValueError):
            gimbal.set_pose(pan_angle=20, tilt_angle=150)
        self.assertEqual(controller.commands, [])

    def test_nonfinite_commands_are_rejected_without_output(self):
        for value in (math.nan, math.inf, -math.inf, True):
            controller=RecordingController(); chassis=RosmasterChassis(controller=controller)
            with self.subTest(value=value), self.assertRaises(ValueError):
                chassis.set_four_wheels(value,0,0,0)
            self.assertEqual(controller.commands, [])

    def test_invalid_ramp_duration_does_not_start_motion(self):
        controller=RecordingController(); chassis=RosmasterChassis(controller=controller)
        for duration in (math.nan, math.inf, -1):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                event=threading.Event(); event.set()
                chassis.ramp_to(10,10,duration,stop_event=event)
        self.assertEqual(controller.commands, [])

    def test_stop_attempts_all_three_zero_writes_and_reports_first_error(self):
        controller=RecordingController(); controller.failures=1
        chassis=RosmasterChassis(controller=controller)
        with patch("vehicle_control.base.time.sleep"), self.assertRaises(OSError):
            chassis.stop()
        self.assertEqual(controller.commands, [(0,0,0,0)]*3)

    def test_close_releases_serial_when_stop_fails(self):
        controller=RecordingController(); controller.failures=3
        with patch("vehicle_control.platforms.rosmaster._new_rosmaster", return_value=controller):
            chassis=RosmasterChassis()
        with patch("vehicle_control.base.time.sleep"), self.assertRaises(OSError):
            chassis.close()
        self.assertTrue(controller.closed)
        self.assertEqual(controller.commands, [(0,0,0,0)]*3)
        chassis.close()

    def test_concurrent_stop_cancels_ramp_without_resuming_target(self):
        controller=RecordingController();chassis=RosmasterChassis(controller=controller)
        stopped=[]
        def sleep(seconds):
            if seconds==.02 and not stopped:
                stopped.append(True);chassis.stop()
        with patch("vehicle_control.base.time.monotonic",side_effect=[0,.02,.12]),patch("vehicle_control.base.time.sleep",side_effect=sleep):
            chassis.ramp_to(10,10,.1)
        first_zero=controller.commands.index((0,0,0,0))
        self.assertTrue(all(not any(x) for x in controller.commands[first_zero:]))
        self.assertEqual(chassis.current_left,0)

    def test_cancelled_ramp_writes_stop_instead_of_leaving_previous_motion(self):
        controller=RecordingController();chassis=RosmasterChassis(controller=controller)
        chassis.set_wheels(10,10);event=threading.Event();event.set()
        with patch("vehicle_control.base.time.sleep"):
            chassis.ramp_to(20,20,.1,stop_event=event)
        self.assertEqual(controller.commands[-1],(0,0,0,0))
        self.assertEqual(chassis.current_left,0)

    def test_public_factories_preserve_explicit_borrowed_session(self):
        controller=RecordingController()
        config=self.profile(motion_calibrated=True,gimbal_calibrated=True)
        config["chassis"]["owns_controller"]=False
        config["gimbal"]={"owns_controller":False}
        chassis=create_chassis(config,controller=controller)
        gimbal=create_gimbal(config,controller=controller)
        chassis.close(stop=False)
        self.assertFalse(controller.closed)
        gimbal.set_pose(20,30);gimbal.close()
        self.assertFalse(controller.closed)
        controller.close()

    def test_emergency_stop_ignores_nonessential_mapping_errors(self):
        config=self.profile(motion_calibrated=False)
        config["chassis"].update({"motor_order":[0,0,2,3],"command_limit":101,"wheel_signs":[1,0,1,1]})
        controller=RecordingController()
        with patch("vehicle_control.platforms.rosmaster._new_rosmaster",return_value=controller),patch("vehicle_control.base.time.sleep"):
            emergency_stop(config)
        self.assertEqual(controller.commands,[(0,0,0,0)]*3)
        self.assertTrue(controller.closed)

    def test_close_rejects_new_motion_while_stop_is_in_progress(self):
        controller=RecordingController();chassis=RosmasterChassis(controller=controller)
        def sleep(seconds):
            with self.assertRaises(RuntimeError):chassis.set_wheels(10,10)
        with patch("vehicle_control.base.time.sleep",side_effect=sleep):chassis.close()
        self.assertTrue(controller.closed)
        self.assertEqual(controller.commands,[(0,0,0,0)]*3)

    def test_borrowed_controller_is_not_closed_by_one_component(self):
        controller=RecordingController()
        chassis=RosmasterChassis(controller=controller, owns_controller=False); gimbal=RosmasterGimbal(controller=controller, owns_controller=False)
        chassis.close(stop=False)
        self.assertFalse(controller.closed)
        gimbal.set_pose(20,30)
        gimbal.close()
        self.assertFalse(controller.closed)
        controller.close()

if __name__ == "__main__":
    unittest.main()

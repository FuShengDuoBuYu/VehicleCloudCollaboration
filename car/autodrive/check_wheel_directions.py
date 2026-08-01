#!/usr/bin/env python3
"""Guarded side or individual-wheel checks for a lifted Raspbot chassis."""

import argparse
import json
from pathlib import Path
import sys
import time


AUTODRIVE_DIR = Path(__file__).resolve().parent
CAR_DIR = AUTODRIVE_DIR.parent
CONTROL_DIR = CAR_DIR / "control"
REPO_ROOT = CAR_DIR.parent
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

CONFIRMATION = "WHEELS_ARE_LIFTED"
WHEEL_LABELS = {
    0: "front-left",
    1: "rear-left",
    2: "front-right",
    3: "rear-right",
}
MOTOR_ID_TO_LOGICAL_INDEX = {
    0: 0,  # front-left
    1: 1,  # rear-left
    2: 2,  # front-right
    3: 3,  # rear-right
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Check wheel direction and relative speed with all wheels physically lifted"
        )
    )
    parser.add_argument(
        "--confirm-wheels-lifted",
        default="",
        help=f"Required exact value: {CONFIRMATION}",
    )
    parser.add_argument("--pwm", type=int, default=12)
    parser.add_argument("--duration", type=float, default=0.4)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--wheel",
        type=int,
        choices=tuple(WHEEL_LABELS),
        help=(
            "Pulse only this motor: 0=front-left, 1=rear-left, "
            "2=front-right, 3=rear-right"
        ),
    )
    target.add_argument(
        "--all-wheels",
        action="store_true",
        help="Pulse all four wheels together at the same PWM",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "reverse"),
        default="forward",
        help="PWM direction for --wheel or --all-wheels mode",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "onboard_runtime" / "wheel_check.json"),
    )
    return parser


def yes_no(prompt):
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


def individual_targets(wheel, pwm, direction):
    """Return logical wheel targets for exactly one physical motor ID."""
    target = int(pwm) if direction == "forward" else -int(pwm)
    targets = [0, 0, 0, 0]
    targets[MOTOR_ID_TO_LOGICAL_INDEX[int(wheel)]] = target
    return tuple(targets)


def run_individual_check(chassis, wheel, pwm, direction, duration):
    label = WHEEL_LABELS[wheel]
    targets = individual_targets(wheel, pwm, direction)
    print(
        f"Pulsing motor {wheel} ({label}) {direction} at PWM {pwm} "
        f"for {duration:.2f}s; targets={targets}",
        flush=True,
    )
    try:
        chassis.set_four_wheels(*targets)
        time.sleep(duration)
    finally:
        chassis.stop()
    print("Stopped all four wheels.", flush=True)


def run_all_wheels_check(chassis, pwm, direction, duration):
    target = int(pwm) if direction == "forward" else -int(pwm)
    targets = (target, target, target, target)
    print(
        f"Pulsing all four wheels {direction} at PWM {pwm} "
        f"for {duration:.2f}s; targets={targets}",
        flush=True,
    )
    try:
        chassis.set_four_wheels(*targets)
        time.sleep(duration)
    finally:
        chassis.stop()
    print("Stopped all four wheels.", flush=True)


def main():
    args = build_parser().parse_args()
    if args.confirm_wheels_lifted != CONFIRMATION:
        raise ValueError(
            f"refusing motor access; pass --confirm-wheels-lifted {CONFIRMATION}"
        )
    if not 1 <= args.pwm <= 30:
        raise ValueError("--pwm must be in [1, 30]")
    if not 0.1 <= args.duration <= 1.0:
        raise ValueError("--duration must be in [0.1, 1.0]")

    from vehicle_control.hardware import RospbotChassis

    chassis = RospbotChassis()
    if args.wheel is not None:
        run_individual_check(
            chassis,
            args.wheel,
            args.pwm,
            args.direction,
            args.duration,
        )
        return 0
    if args.all_wheels:
        run_all_wheels_check(
            chassis,
            args.pwm,
            args.direction,
            args.duration,
        )
        return 0

    observations = {}
    try:
        input("Confirm the vehicle is supported and all four wheels are clear. Press Enter.")
        for side, command in (
            ("left", (args.pwm, 0)),
            ("right", (0, args.pwm)),
        ):
            input(f"Ready to pulse the {side} wheels. Press Enter.")
            chassis.set_wheels(*command)
            time.sleep(args.duration)
            chassis.stop()
            observations[f"{side}_positive_moves_forward"] = yes_no(
                f"Did positive PWM move the {side} wheels in the vehicle-forward direction?"
            )
    finally:
        chassis.stop()

    payload = {
        "pwm": args.pwm,
        "duration": args.duration,
        "observations": observations,
        "suggested_left_sign": (
            1 if observations.get("left_positive_moves_forward") else -1
        ),
        "suggested_right_sign": (
            1 if observations.get("right_positive_moves_forward") else -1
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")
    print("Copy the suggested signs into onboard_runtime.yaml, then repeat dry-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

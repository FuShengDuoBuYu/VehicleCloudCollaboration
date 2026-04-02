# path_follow

This app is a sibling of train_car for recording a path and replaying it with Donkeycar path-follow autopilot.

## Run

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car/control/path_follow
python manage.py drive
```

## Workflow

1. Use web steering and throttle in User mode to drive a straight path.
2. Press web/w1 to save path.
3. Switch mode to Local and let the car follow the saved path.

## Web Buttons

1. web/w1: Save path
2. web/w2: Load path
3. web/w3: Reset origin
4. web/w4: Erase path
5. web/w5: Toggle recording

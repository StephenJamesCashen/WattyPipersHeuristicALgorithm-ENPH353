from pynput.keyboard import Key, Listener
import anki_vector as av
import datetime
import time
import numpy as np
from anki_vector.util import degrees
import cv2 as cv

left_wheel_speed = 0
right_wheel_speed = 0

left_flag = 0
right_flag = 0
back_flag = 0
for_flag = 0
shift_flag = False

speed = 100


def image_capture(robot, display_time, display_screen, display_comp, save):
    image = robot.camera.capture_single_image()
    if display_screen:
        screen = image.raw_image.resize((184, 96))
        screen = av.screen.convert_image_to_screen_data(screen)
        robot.screen.set_screen_with_image_data(screen, display_time, interrupt_running=False)
    if display_comp:
        img = np.array(image.raw_image.convert('RGB'))[:, :, ::-1].copy()
        cv.imshow("image", img)
        cv.waitKey(100)
    if save:
        image.raw_image.save("..\\..\\Image_Data\\image_" + str(int(time.time())) + ".png")


def main():
    # Modify the SN to match your robot’s SN
    ANKI_SERIAL = '0090452f'

    # Override walk around default behaviour 
    ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

    # Construct robot object, connect via serial number. Seems odd? Shouldn't it conncet with the IP address?
    with av.Robot(serial=ANKI_SERIAL,
                  behavior_control_level=ANKI_BEHAVIOR) as robot:
        print("Connected")
        # Send say_text to robot
        verbose = 0
        capture_mode = "on_key"
        period = 0.5
        display_comp = 1
        display_screen = 0
        save = 1

        if verbose:
            robot.behavior.say_text("Connected!")

        def on_press(key):
            global left_flag, right_flag, for_flag, back_flag, shift_flag

            if key == Key.up:
                for_flag = 1

            elif key == Key.down:
                back_flag = 1

            elif key == Key.left:
                left_flag = 1

            elif key == Key.right:
                right_flag = 1

            left_wheel_speed = speed * (for_flag + right_flag - left_flag - back_flag) * (1 + shift_flag)
            right_wheel_speed = speed * (for_flag - right_flag + left_flag - back_flag) * (1 + shift_flag)

            robot.motors.set_wheel_motors(left_wheel_speed, right_wheel_speed)

        def on_release(key):
            global left_flag, right_flag, for_flag, back_flag, shift_flag

            if key == Key.up:
                for_flag = 0

            elif key == Key.down:
                back_flag = 0

            elif key == Key.left:
                left_flag = 0

            elif key == Key.right:
                right_flag = 0

            elif key == Key.space and capture_mode == "on_key":
                image_capture(robot, 5, display_screen, display_comp, save)
            elif key == Key.shift or Key.shift_r:
                shift_flag = not shift_flag

            left_wheel_speed = speed * (for_flag + right_flag - left_flag - back_flag) * (1 + shift_flag)
            right_wheel_speed = speed * (for_flag - right_flag + left_flag - back_flag) * (1 + shift_flag)

            robot.motors.set_wheel_motors(left_wheel_speed, right_wheel_speed)
            if key == Key.esc:
                # Stop listener
                return False

        with Listener(on_press=on_press,
                      on_release=on_release) as listener:
            i = 0

            while listener.running:
                robot.behavior.set_head_angle(degrees(0))
                robot.behavior.set_lift_height(1.0)

                if capture_mode == "periodic":
                    image_capture(robot, period, display_screen, display_comp, save)

                time.sleep(period)

        if verbose:
            robot.behavior.say_text("Done!")
        else:
            print("done!")
        while True:
            time.sleep(10)


if __name__ == "__main__":
    main()

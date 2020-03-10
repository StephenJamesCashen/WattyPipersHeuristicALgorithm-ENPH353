from pynput.keyboard import Key, Listener
import anki_vector as av
import datetime
import time
from anki_vector.util import degrees

left_wheel_speed = 0 
right_wheel_speed = 0

left_flag = 0
right_flag = 0
back_flag = 0
for_flag = 0
shift_flag = False

speed = 100

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

            elif key == Key.shift or Key.shift_r:
                shift_flag = not shift_flag

            left_wheel_speed = speed*(for_flag+right_flag-left_flag-back_flag) * (1 + shift_flag)
            right_wheel_speed = speed*(for_flag-right_flag+left_flag-back_flag) * (1 + shift_flag)

            robot.motors.set_wheel_motors(left_wheel_speed, right_wheel_speed)
            if key == Key.esc:
                # Stop listener
                return False

        with Listener(on_press=on_press,
                    on_release=on_release) as listener:
            robot.camera.init_camera_feed()
            
            i = 0
            while listener.running:
                robot.behavior.set_head_angle(degrees(0))

                image = robot.camera.latest_image
                image.raw_image.save("Competition\\Image_Data\\image_" + str(int(time.time())) + ".png")

                time.sleep(1)

        robot.behavior.say_text("Done!")

if __name__ == "__main__":
	main()

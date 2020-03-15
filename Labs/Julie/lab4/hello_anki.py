
import anki_vector as av

def main():
    # Modify the SN to match your robot’s SN
    ANKI_SERIAL = '0090452f'
    ANKI_BEHAVIOR = \
        av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

    with av.Robot(serial=ANKI_SERIAL,
                  behavior_control_level=ANKI_BEHAVIOR) as robot:
        print("Say 'Hello World'...")
        robot.behavior.say_text("Extermine ate. Extermine ate. Extermine ate.")


if __name__ == "__main__":
    main()

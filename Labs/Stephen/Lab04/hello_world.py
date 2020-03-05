import anki_vector as av


def main():
    # Modify the SN to match your robot’s SN
    ANKI_SERIAL = '0090452f'
    
    # Override walk around default behaviour 
    ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

    # Construct robot object, connect via serial number. Seems odd? Shouldn't it conncet with the IP address?
    with av.Robot(serial=ANKI_SERIAL,
                  behavior_control_level=ANKI_BEHAVIOR) as robot:
        print("Introduce yourself...")

        # Send say_text to robot
        robot.behavior.say_text("Hi Adam it is nice to meet you!")


if __name__ == "__main__":
	main()

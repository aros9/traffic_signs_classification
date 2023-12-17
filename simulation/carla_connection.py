import carla
import os

try:
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # client.load_world('Town02')

    opendrive_path = os.getcwd() + '\\config.xodr'

    with open(opendrive_path, 'r') as file:
        opendrive_content = file.read()

    client.set_timeout(30.0)  # Set a longer timeout (30s)

    client.generate_opendrive_world(opendrive_content)

except Exception as e:
    print(f"An error occurred: {e}")

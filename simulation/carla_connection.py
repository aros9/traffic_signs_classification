import carla
import os

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# client.load_world('Town05')

opendrive_path = os.getcwd() + '\\config.xodr'

with open(opendrive_path, 'r') as file:
    opendrive_content = file.read()

client.set_timeout(10.0)  # Set a longer timeout (10s)

world = client.generate_opendrive_world(opendrive_content)


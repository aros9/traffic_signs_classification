import carla
import random
import time
import cv2
import numpy as np
import keyboard

def main():
    # Connect to the Carla server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        # Load the Carla world
        world = client.get_world()

        # Get the Carla blueprint library
        blueprint_library = world.get_blueprint_library()

        # Choose a random vehicle blueprint
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))

        # Set the spawn point for the vehicle
        spawn_point = random.choice(world.get_map().get_spawn_points())

        # Spawn the vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        vehicle.set_autopilot(True)

        # Choose a camera sensor blueprint
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')

        # Set the relative transform for the camera
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # Attach the camera sensor to the vehicle
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Register a callback to receive the camera data
        camera_sensor.listen(lambda image: process_image(image))

        # Wait for the user to press the 'ESC' key to exit
        print("Press 'ESC' to exit.")
        keyboard.wait('esc')

    finally:
        # Destroy actors when done
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'camera_sensor' in locals():
            camera_sensor.destroy()


def process_image(image):
    # Convert the raw image data to a numpy array
    img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img_array = np.reshape(img_array, (image.height, image.width, 4))

    # Remove the alpha channel
    img_array = img_array[:, :, :3]

    # Convert to BGR (OpenCV uses BGR instead of RGB)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    # Display the image in real-time
    cv2.imshow("Camera Feed", img_bgr)
    cv2.waitKey(1)  # Add a short delay to allow the image to be displayed


if __name__ == '__main__':
    main()

import carla
import random
import time
import cv2
import numpy as np
import keyboard
import tensorflow as tf

detection_model = tf.keras.models.load_model('../saved_models/detection_model')


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
        camera_bp.set_attribute('image_size_x', '200')
        camera_bp.set_attribute('image_size_y', '200')
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

    # Convert to grayscale
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Resize the image to 200x200
    resized_img = cv2.resize(gray_img, (200, 200))

    # Normalize the pixel values to be between 0 and 1
    normalized_img = resized_img / 255.0

    # Expand dimensions to match the model's input shape (add batch dimension)
    input_img = np.expand_dims(normalized_img, axis=0)

    # Perform inference with your model (replace with your actual model inference code)
    model_output = detection_model.predict(input_img)

    # Print or process the model output as needed
    print("Model Output:", model_output)


if __name__ == '__main__':
    main()

from PIL import Image, ImageDraw
import os
import random

# Path to the directory containing your traffic sign images
traffic_signs_path = os.getcwd() + "/dataset/A-1"

# Path to the directory containing your background images
backgrounds_path = os.getcwd() + "/backgrounds"

# Output directory for the new dataset
output_path = os.getcwd() + "/test"

# Number of images to generate
num_images = 100


def overlay_traffic_sign(background: Image, traffic_sign: Image, x: int, y: int):
    """
    Function that overlays traffic sign image on the background.

    :param background: Image of background to be used.
    :param traffic_sign:  Image of traffic sign to be used.
    :param x: Position x to overlay image.
    :param y: Position y to overlay image.
    """
    # Resize traffic sign to a random scale (you can customize this)
    scale_factor = random.uniform(0.5, 1.5)
    traffic_sign = traffic_sign.resize((int(traffic_sign.width * scale_factor), int(traffic_sign.height * scale_factor)))

    # Rotate traffic sign to a random angle (you can customize this)
    rotation_angle = random.uniform(-30, 30)
    traffic_sign = traffic_sign.rotate(rotation_angle, expand=True)

    # Ensure traffic sign has an alpha channel (transparency)
    if traffic_sign.mode != 'RGBA':
        traffic_sign = traffic_sign.convert('RGBA')

    # Composite the images
    background.paste(traffic_sign, (x, y), traffic_sign)

    # Draw bounding box
    draw = ImageDraw.Draw(background)
    draw.rectangle([x, y, x + traffic_sign.width, y + traffic_sign.height], outline="red", width=2)

    return background


# Generate dataset
def main():
    for i in range(num_images):
        # Load random background image
        background_file = random.choice(os.listdir(backgrounds_path))
        background_path = os.path.join(backgrounds_path, background_file)
        background = Image.open(background_path)

        # Load random traffic sign image
        traffic_sign_file = random.choice(os.listdir(traffic_signs_path))
        traffic_sign_path = os.path.join(traffic_signs_path, traffic_sign_file)
        traffic_sign = Image.open(traffic_sign_path)

        # Random position to overlay traffic sign on background
        x = random.randint(0, background.width - traffic_sign.width)
        y = random.randint(0, background.height - traffic_sign.height)

        # Overlay traffic sign on background
        image_with_traffic_sign = overlay_traffic_sign(background.copy(), traffic_sign, x, y)

        # Save the image with annotation in the filename
        output_file = f"x{x}_y{y}.png"
        output_file_path = os.path.join(output_path, output_file)
        image_with_traffic_sign.save(output_file_path)


if __name__ == "__main__":
    main()
    print("Dataset generation complete.")

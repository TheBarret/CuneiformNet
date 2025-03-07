import os
import json
import shutil
import random
from PIL import Image, ImageDraw, ImageFont, ImageChops

# Config Switch: Local or Colab
ENVIRONMENT = "local"

# Configuration Settings
SETTINGS = {
    "local": {
        "IMAGE_SIZE": (256, 256),
        "FONT_PATH": "NotoSansCuneiform-Regular.ttf",
        "OUTPUT_IMAGE_DIR": "images",
        "OUTPUT_JSON_PATH": "cuneiform.json",
        "CUNEIFORM_RAW": "cuneiform_desc.txt",
        "CUNEIFORM_START": 0x12000,
        "CUNEIFORM_END": 0x12000 + 2,  # + 2 for debugging
        "AUGMENTATIONS": 4,
        "FONT_SIZE": 100,
        "FONT_OFFSET_Y": 70,          # Control the x-axis of the font center
        "FONT_OFFSET_X": 10,          # Control the y-axis of the font center
        "ROTATION_RANGE": (-45, 45),  # Rotational noise (Degrees)
        "NOISE_LEVEL": 15,            # Pixel intensity noise
    },
    "colab": {
        "IMAGE_SIZE": (128, 128),
        "FONT_PATH": "/content/NotoSansCuneiform-Regular.ttf",
        "OUTPUT_IMAGE_DIR": "images",
        "OUTPUT_JSON_PATH": "cuneiform.json",
        "CUNEIFORM_RAW": "/content/cuneiform_desc.txt",
        "CUNEIFORM_START": 0x12000,
        "CUNEIFORM_END": 0x12000 + 0x400,  # + 1024 for production
        "AUGMENTATIONS": 4,           # Extra augmented variations
        "FONT_SIZE": 100,             # Font size
        "FONT_OFFSET_Y": 70,          # Control the x-axis of the font center
        "FONT_OFFSET_X": 10,          # Control the y-axis of the font center
        "ROTATION_RANGE": (-45, 45),  # Rotational noise (Degrees)
        "NOISE_LEVEL": 15,            # Pixel intensity noise
    }
}

# Select the appropriate settings based on the environment
config = SETTINGS[ENVIRONMENT]
os.makedirs(config["OUTPUT_IMAGE_DIR"], exist_ok=True)

def fetch_symbol_description(code_point):
    try:
        with open(config["CUNEIFORM_RAW"], 'r', encoding='utf-8') as file:
            for line in file:
                symbol, _, description = line.partition('=')
                if int(symbol, 16) == code_point:
                    return description.strip()
    except FileNotFoundError:
        print(f"Error: {config['CUNEIFORM_RAW']} not found")
    return f"Cuneiform U+{code_point:04X}"

def add_noise(image):
    noise = Image.effect_noise(config["IMAGE_SIZE"], config["NOISE_LEVEL"])
    return ImageChops.add(image, noise)

def rotate_image(image):
    width, height = image.size
    center_x, center_y = width // 2, height // 2
    rotated_image = image.rotate(
        random.randint(*config["ROTATION_RANGE"]), # Random angle within the range
        center=(center_x, center_y),               # Rotate around the center
        expand=True,                               # Expand the canvas to fit the rotated image
        fillcolor="white"                          # Fill the background with white
    )

    # crop rollback
    rotated_width, rotated_height = rotated_image.size
    left = (rotated_width - width) // 2
    top = (rotated_height - height) // 2
    right = left + width
    bottom = top + height

    cropped_image = rotated_image.crop((left, top, right, bottom))

    return cropped_image

def generate_cuneiform_images():
    """Generate Cuneiform images with augmentations and save metadata."""
    truth_table = {}
    font = ImageFont.truetype(config["FONT_PATH"], config["FONT_SIZE"])

    for code_point in range(config["CUNEIFORM_START"], config["CUNEIFORM_END"] + 1):
        try:
            symbol = chr(code_point)
            base_image = Image.new("L", config["IMAGE_SIZE"], "white")
            draw = ImageDraw.Draw(base_image)

            bbox = draw.textbbox((0, 0), symbol, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x_1 = (config["IMAGE_SIZE"][0] - text_width) // 2 - config["FONT_OFFSET_X"]
            y_1 = (config["IMAGE_SIZE"][1] - text_height) // 2 - config["FONT_OFFSET_Y"]
            x, y = x_1, y_1

            draw.text((x, y), symbol, font=font, fill="black")

            # Save Original
            filename = f"cuneiform_{code_point:04X}_0.png"
            base_image.save(os.path.join(config["OUTPUT_IMAGE_DIR"], filename))

            # Store JSON metadata
            truth_table[filename] = {
                "unicode": f"U+{code_point:04X}",
                "symbol": symbol,
                "code": code_point,
                "description": fetch_symbol_description(code_point),
                "category": "Cuneiform",
                "type": "original",
            }

            # Save Augmented Variations
            for i in range(config["AUGMENTATIONS"]):
                aug_img = add_noise(rotate_image(base_image))
                aug_filename = f"cuneiform_{code_point:04X}_{i+1}.png"
                aug_img.save(os.path.join(config["OUTPUT_IMAGE_DIR"], aug_filename))
                truth_table[aug_filename] = {**truth_table[filename], "type": f"augmented_{i}"}

        except Exception as e:
            print(f"Error generating U+{code_point:04X}: {e}")

    # Save JSON
    with open(config["OUTPUT_JSON_PATH"], "w", encoding="utf-8") as f:
        json.dump(truth_table, f, ensure_ascii=False, indent=4)
    try:
        fn = os.path.join(config["OUTPUT_IMAGE_DIR"], config["OUTPUT_JSON_PATH"])
        if os.path.isfile(fn):
            os.remove(fn)
        shutil.move(config["OUTPUT_JSON_PATH"], config["OUTPUT_IMAGE_DIR"])
    except Exception as e:
        print(f"Error moving dataset file: {e}")

    print(f"Generated {len(truth_table)} augmented Cuneiform symbols.")

# ─── RUN ──────────────────────────────────────────────
if __name__ == "__main__":
    generate_cuneiform_images()

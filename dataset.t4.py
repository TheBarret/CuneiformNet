import os
import json
import shutil
import requests
from PIL import Image, ImageDraw, ImageFont

# Global settings
IMAGE_SIZE          = (64, 64)
FONT_PATH           = "/content/NotoSansCuneiform-Regular.ttf"
OUTPUT_IMAGE_DIR    = "images"
OUTPUT_JSON_PATH    = "cuneiform.json"
CUNEIFORM_RAW       = "/content/cuneiform_desc.txt"
CUNEIFORM_START     = 0x12000
CUNEIFORM_END       = 0x123FF
#CUNEIFORM_END       = CUNEIFORM_START + 25
FONT_SIZE           = 35
FONT_OFFSET_Y       = 40

# Ensure output directory exists
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

def fetch_symbol_description(code_point, file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                symbol, _, description = line.partition('=')
                if int(symbol, 16) == code_point:
                    return description.strip()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return f"Cuneiform symbol U+{code_point:04X}"

# Function to generate centered Cuneiform symbol images
def generate_cuneiform_images():
    truth_table = {}
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    for code_point in range(CUNEIFORM_START, CUNEIFORM_END + 1):
        try:
            symbol = chr(code_point)
            image = Image.new("L", IMAGE_SIZE, "white")
            draw = ImageDraw.Draw(image)

            # Get text bounding box
            bbox = draw.textbbox((0, 0), symbol, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate centered position
            x = (IMAGE_SIZE[0] - text_width) // 2
            y = (IMAGE_SIZE[1] - text_height) - FONT_OFFSET_Y

            # Draw symbol
            draw.text((x, y), symbol, font=font, fill="black")

            # Save image
            filename = f"cuneiform_{code_point:04X}.png"
            image_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
            image.save(image_path)

            # Fetch description
            description = fetch_symbol_description(code_point, CUNEIFORM_RAW)

            # Store JSON entry
            truth_table[filename] = {
                "unicode": f"U+{code_point:04X}",
                "symbol": symbol,
                "description": description,
                "category": "Cuneiform",
            }
        except Exception as e:
            print(f"Error generating symbol for U+{code_point:04X}: {e}")

    # Save JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(truth_table, f, ensure_ascii=False, indent=4)
    try:
        fn = '/content/' + OUTPUT_IMAGE_DIR + '/' + OUTPUT_JSON_PATH
        if os.path.isfile(fn):
            os.remove(fn)
        shutil.move('/content/' + OUTPUT_JSON_PATH, '/content/images')
        print(f"Dataset moved to {fn}")
    except FileNotFoundError:
        print(f"Dataset file {fn} not found")
    except Exception as e:
        print(f"Error moving dataset file: {e}")
    print(f"Generated {len(truth_table)} Cuneiform symbols and truth table.")

# Main function
if __name__ == "__main__":
    generate_cuneiform_images()
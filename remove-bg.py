import os
from rembg import remove
from PIL import Image
import io

# Define your paths (Use absolute paths for Unity integration)
INPUT_FOLDER = "/Users/shub/rack/code-rack/playground/hollow/input"
OUTPUT_FOLDER = "/Users/shub/rack/code-rack/playground/hollow/output"
def process_folder():
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.png")

            # Skip if already processed to save time
            if os.path.exists(output_path):
                continue

            print(f"Processing: {filename}")
            
            try:
                with open(input_path, 'rb') as i:
                    input_data = i.read()
                    # This is the core local AI call
                    output_data = remove(input_data) 
                    
                with open(output_path, 'wb') as o:
                    o.write(output_data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "_main_":
    process_folder()
    print("Batch processing complete.")

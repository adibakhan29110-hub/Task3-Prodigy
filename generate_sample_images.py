from PIL import Image, ImageDraw

import os

# Create folder if not exists
os.makedirs("data/sample_pairs", exist_ok=True)

# Function to create a colored rectangle image
def create_image(color, size=(128,128)):
    img = Image.new("RGB", size, color)
    return img

# Generate 3 side-by-side images
colors = [("red","blue"), ("green","yellow"), ("purple","orange")]

for i, (left_color, right_color) in enumerate(colors, start=1):
    left = create_image(left_color)
    right = create_image(right_color)
    
    # Side-by-side
    combined = Image.new("RGB", (left.width + right.width, left.height))
    combined.paste(left, (0,0))
    combined.paste(right, (left.width,0))
    
    # Save
    path = f"data/sample_pairs/sample{i}.jpg"
    combined.save(path)
    print(f"Saved {path}")

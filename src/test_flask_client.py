# src/test_flask_client.py
import requests
from PIL import Image
from io import BytesIO

# Flask API URL
url = "http://127.0.0.1:5000/predict"

# Image to send
image_path = "data/sample_pairs/sample1.jpg"

with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    output_path = data["output"]
    print(f"Output saved at: {output_path}")

    # Open image directly in VSCode (or default image viewer)
    img = Image.open(output_path)
    img.show()
else:
    print(f"Error: {response.text}")

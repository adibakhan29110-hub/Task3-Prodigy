# src/demo_flask_api.py
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from models import GeneratorUNet
import os

app = Flask(__name__)

# -------------------------
# Device and model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = GeneratorUNet().to(device)
checkpoint = "checkpoints/latest.pth"
checkpoint_data = torch.load(checkpoint, map_location=device)
gen.load_state_dict(checkpoint_data['model_state_dict'])
gen.eval()

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# -------------------------
# API route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error":"No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file).convert("RGB")

    # Crop left-half if side-by-side
    w,h = img.size
    if w > h:
        w2 = int(w/2)
        inp = img.crop((0,0,w2,h))
    else:
        inp = img

    inp_tensor = transform(inp).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = gen(inp_tensor)
    out_tensor = (out_tensor + 1)/2.0

    # Save result
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/flask_result.png"
    transforms.ToPILImage()(out_tensor.squeeze(0).cpu()).save(out_path)

    return jsonify({"output": out_path})

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)

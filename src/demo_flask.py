from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
from models import GeneratorUNet
import io

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = GeneratorUNet().to(device)

try:
    gen.load_state_dict(torch.load('../checkpoints/latest.pth', map_location=device)["model_state"])
    print("Model loaded successfully.")
except:
    print("ERROR: Could not load checkpoint.")

gen.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = gen(inp)

    fake = fake * 0.5 + 0.5  # unnormalize
    fake_img = transforms.ToPILImage()(fake.squeeze())

    buf = io.BytesIO()
    fake_img.save(buf, format='PNG')
    buf.seek(0)

    return buf.getvalue(), 200, {'Content-Type': 'image/png'}


if __name__ == '__main__':
    app.run(debug=True)

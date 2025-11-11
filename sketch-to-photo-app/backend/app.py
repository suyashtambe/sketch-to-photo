import os
import io
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn

# ==== CONFIG ====
MODEL_WEIGHTS = r"C:\Users\Suyash Tambe\Desktop\sketch-photo\pix2pix_outputs\generator_epoch3.pth"
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ==== MODEL ====
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.down = nn.Sequential(
            self.contract(in_channels, features),
            self.contract(features, features * 2),
            self.contract(features * 2, features * 4),
            self.contract(features * 4, features * 8),
            self.contract(features * 8, features * 8),
        )
        self.up = nn.Sequential(
            self.expand(features * 8, features * 8),
            self.expand(features * 16, features * 4),
            self.expand(features * 8, features * 2),
            self.expand(features * 4, features),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def contract(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def expand(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)
        skips = skips[:-1][::-1]  # reverse except bottleneck
        for i, layer in enumerate(self.up):
            x = layer(x)
            if i < len(skips):
                x = torch.cat([x, skips[i]], dim=1)
        return self.final(x)

# 3) Build and load weights
gen = UNetGenerator()
state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
gen.load_state_dict(state_dict)
gen.to(DEVICE)
gen.eval()

# 4) Preprocessing: make each sketch into a 3‑channel tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),                    # [0–1]
    transforms.Normalize([0.5], [0.5]),       # [-1,1]
    transforms.Lambda(lambda x: x.repeat(3,1,1))  # 1→3 channels
])
# ==== FLASK ====
app = Flask(__name__)

from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/generate', methods=['POST'])
def generate():
    if 'sketch' not in request.files:
        return jsonify({'error': 'No sketch file provided'}), 400

    file = request.files['sketch']
    img = Image.open(file.stream).convert("L")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = gen(tensor)
        output = (output.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)

    # Convert to PIL for returning
    output_image = transforms.ToPILImage()(output)

    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/')
def index():
    return "Sketch-to-Photo GAN Inference API"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

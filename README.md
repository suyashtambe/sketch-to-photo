# 🎨 Sketch to Photo — GAN-Based Image Translation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🧠 Project Overview

**Sketch-to-Photo** is a GANs project that converts hand-drawn sketches into **realistic photos** using the **Pix2Pix Conditional GAN (cGAN)** architecture.  
It demonstrates the potential of Generative Adversarial Networks (GANs) in **image-to-image translation** tasks.

---

## 🧩 Architecture

- **Model:** Pix2Pix (Conditional GAN)  
- **Generator:** U-Net  
- **Discriminator:** PatchGAN  
- **Framework:** PyTorch  
- **Frontend:** React.js  
- **Backend:** Flask (Python API for inference)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/suyashtambe/sketch-to-photo.git
cd sketch-to-photo
2️⃣ Frontend Setup (React)
bash
Copy code
cd sketch-to-photo-app
npm install
npm start
The frontend will start on 👉 http://localhost:3000

3️⃣ Backend Setup (Flask API)
bash
Copy code
cd backend
python app.py
The backend will start on 👉 http://localhost:5000

🧰 Requirements
🐍 Python Dependencies
nginx
Copy code
Flask
torch
torchvision
Pillow
numpy
⚛️ Frontend Dependencies
nginx
Copy code
react
axios
react-router-dom
🧪 How It Works
User uploads or draws a sketch in the React web app.

The image is sent to the Flask backend via REST API.

The backend loads the trained Pix2Pix GAN model and generates a realistic photo.

The output image is displayed instantly in the frontend interface.

📁 Folder Structure
csharp
Copy code
SKETCH-PHOTO/
│
├── coding/                     # GAN training and preprocessing
│   └── training.ipynb
│
├── sketch-to-photo-app/        # Web app (React + Flask)
│   ├── backend/                # Flask backend
│   ├── src/                    # React frontend code
│   └── public/                 # Static assets
│
├── .gitignore
└── README.md
🖼️ Example Results
Input Sketch	Generated Photo

(Replace the placeholders with your actual before/after outputs once ready.)

🚀 Future Enhancements
✏️ Add interactive sketch canvas in frontend

🎯 Improve model with attention-based GANs

☁️ Deploy on cloud (Render / HuggingFace Spaces / AWS)

🧠 Train on larger sketch-photo datasets

🧑‍💻 Author
Suyash Tambe
💡 Deep Learning & AI Enthusiast | Computer Vision | Generative Models
🔗 GitHub Profile

🧾 References
Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)

PyTorch Documentation

Flask Documentation

React Documentation

🪄 License
This project is open-sourced under the MIT License.
You are free to use, modify, and distribute it with attribution.

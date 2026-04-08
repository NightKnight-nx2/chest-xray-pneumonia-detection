import * as ort from 'onnxruntime-web';


// --- Carousel Logic ---
const slides = document.querySelectorAll('.slide');
let currentSlide = 0;

function showSlide(index) {
    slides.forEach(s => s.classList.remove('active'));
    
    if (index >= slides.length) currentSlide = 0;
    else if (index < 0) currentSlide = slides.length - 1;
    else currentSlide = index;

    slides[currentSlide].classList.add('active');
}

document.getElementById('prev-btn').addEventListener('click', () => showSlide(currentSlide - 1));
document.getElementById('next-btn').addEventListener('click', () => showSlide(currentSlide + 1));

// Auto rotate every 5s
setInterval(() => {
    showSlide(currentSlide + 1);
}, 5000);

// --- ONNX Inference Logic ---
let session = null;
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');
const confidenceFill = document.getElementById('confidence-fill');
const resultBox = document.getElementById('result-box');
const processCanvas = document.getElementById('process-canvas');
const ctx = processCanvas.getContext('2d');
const imagePreview = document.getElementById('image-preview');

async function loadModel() {
    try {
        console.log("Loading model...");
        predictionText.textContent = "Loading AI Model...";
        resultBox.style.display = "block";
        
        // Use absolute path for Vite /public
        session = await ort.InferenceSession.create('/model/model.onnx');
        console.log("Model loaded successfully.");
        
        resultBox.style.display = "none";
    } catch (e) {
        console.error("Failed to load model:", e);
        predictionText.textContent = "Failed to load model.";
    }
}
loadModel();

// Image Upload Handling
const fileInput = document.getElementById('image-upload');
const uploadArea = document.getElementById('upload-area');

fileInput.addEventListener('change', handleFile);
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFile();
    }
});

function handleFile() {
    if (!fileInput.files.length) return;
    const file = fileInput.files[0];
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'inline-block';
        
        imagePreview.onload = () => {
            if(session) {
                runInference(imagePreview);
            } else {
                predictionText.textContent = "Model not ready yet.";
                resultBox.style.display = "block";
            }
        };
    };
    reader.readAsDataURL(file);
}

// Inference Execution
async function runInference(img) {
    resultBox.style.display = "block";
    resultBox.className = "result-box";
    predictionText.textContent = "Analyzing X-Ray...";
    confidenceFill.style.width = "0%";
    confidenceText.textContent = "Processing...";

    // Draw and Resize
    ctx.drawImage(img, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224).data;
    
    // Convert to Float32Array and normalize for PyTorch ResNet50
    const float32Data = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // ONNX expects shape [1, 3, 224, 224] natively flattened (CHW)
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < 224 * 224; i++) {
            let val = imageData[i * 4 + c] / 255.0; // scale 0-1
            float32Data[c * 224 * 224 + i] = (val - mean[c]) / std[c];
        }
    }

    try {
        const tensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
        const results = await session.run({ 'input': tensor });
        const output = results['output'].data; // Logit value

        // Sigmoid
        const probability = 1 / (1 + Math.exp(-output[0]));
        const isPneumonia = probability > 0.5;

        // UI Updates
        let confPercent = isPneumonia ? (probability * 100) : ((1 - probability) * 100);
        confPercent = confPercent.toFixed(1);

        resultBox.classList.add(isPneumonia ? 'pneumonia' : 'normal');
        predictionText.textContent = isPneumonia ? "Pneumonia Detected" : "Normal (Healthy)";
        confidenceText.textContent = `Confidence: ${confPercent}%`;
        
        // Timeout for animation effect
        setTimeout(() => {
            confidenceFill.style.width = `${confPercent}%`;
        }, 100);

    } catch (e) {
        console.error(e);
        predictionText.textContent = "Error occurred during inference.";
    }
}

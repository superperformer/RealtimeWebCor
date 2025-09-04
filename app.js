
import { createWorker } from 'tesseract.js';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ocrOverlay = document.getElementById('ocr-overlay');
const statusMessage = document.getElementById('status-message');
const outputText = document.getElementById('output-text');
const context = canvas.getContext('2d');

let isRecognizing = false;
let lastOCRTime = 0;
const ocrInterval = 400; // ms
const recentResults = [];
const maxRecent = 5;

const VALID_PREFIXES = ['AKE', 'PMC', 'RKN', 'LD3', 'LD7'];
const VALID_SUFFIXES = ['CI', 'BR', 'CX', 'JL', 'NH'];
const CONFUSION_MAP = { '0': 'O', 'O': '0', '1': 'I', 'I': '1', '5': 'S', 'S': '5', '8': 'B', 'B': '8', '2': 'Z', 'Z': '2', '6': 'G', 'G': '6', 'D': '0' };

const worker = await createWorker({
  logger: m => console.log(m),
});

await worker.loadLanguage('eng');
await worker.initialize('eng');
await worker.setParameters({
  tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
  tessedit_pageseg_mode: '7',
  tessedit_ocr_engine_mode: '1',
});

function preprocessImage(imageData) {
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  tempCanvas.width = imageData.width;
  tempCanvas.height = imageData.height;
  tempCtx.putImageData(imageData, 0, 0);

  // Apply contrast and brightness
  tempCtx.filter = 'contrast(200%) brightness(120%)';
  tempCtx.drawImage(tempCanvas, 0, 0);

  return tempCanvas;
}

function extractROI() {
  const roiCanvas = document.createElement('canvas');
  const roiCtx = roiCanvas.getContext('2d');
  const roiWidth = canvas.width * 0.9;
  const roiHeight = canvas.height * 0.4;
  const roiX = canvas.width * 0.05;
  const roiY = canvas.height * 0.3;

  roiCanvas.width = roiWidth;
  roiCanvas.height = roiHeight;
  roiCtx.drawImage(canvas, roiX, roiY, roiWidth, roiHeight, 0, 0, roiWidth, roiHeight);

  return roiCanvas;
}

function applyConfusionMap(text) {
  return text.split('').map(c => CONFUSION_MAP[c] || c).join('');
}

function validateCode(code) {
  const regex = /^[A-Z]{3}\d{5}[A-Z]{2}$/;
  if (!regex.test(code)) return false;

  const prefix = code.slice(0, 3);
  const suffix = code.slice(-2);
  return VALID_PREFIXES.includes(prefix) && VALID_SUFFIXES.includes(suffix);
}

function voteResult(results) {
  const counts = {};
  results.forEach(r => {
    counts[r] = (counts[r] || 0) + 1;
  });
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

async function processFrame() {
  const now = Date.now();
  if (now - lastOCRTime < ocrInterval || isRecognizing) {
    requestAnimationFrame(processFrame);
    return;
  }

  isRecognizing = true;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const roiCanvas = extractROI();
  const imageData = roiCanvas.getContext('2d').getImageData(0, 0, roiCanvas.width, roiCanvas.height);
  const processedCanvas = preprocessImage(imageData);

  try {
    const { data: { text } } = await worker.recognize(processedCanvas);
    const cleanedText = text.replace(/\s/g, '').toUpperCase();
    const candidates = cleanedText.match(/[A-Z]{3}\d{5}[A-Z]{2}/g) || [];

    for (let code of candidates) {
      if (validateCode(code)) {
        recentResults.push(code);
        if (recentResults.length > maxRecent) recentResults.shift();
        const voted = voteResult(recentResults);
        outputText.textContent = voted;
        break;
      } else {
        const corrected = applyConfusionMap(code);
        if (validateCode(corrected)) {
          recentResults.push(corrected);
          if (recentResults.length > maxRecent) recentResults.shift();
          const voted = voteResult(recentResults);
          outputText.textContent = voted;
          break;
        }
      }
    }
  } catch (err) {
    console.error('OCR failed:', err);
  }

  lastOCRTime = now;
  isRecognizing = false;
  requestAnimationFrame(processFrame);
}

async function startCamera() {
  statusMessage.textContent = '正在請求相機權限...';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      statusMessage.textContent = '相機已啟動。';
      requestAnimationFrame(processFrame);
    };
  } catch (err) {
    statusMessage.textContent = '無法存取相機，請檢查權限設定。';
    console.error('無法取得相機串流:', err);
  }
}

window.addEventListener('load', startCamera);

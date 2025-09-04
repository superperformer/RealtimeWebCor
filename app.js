// app.js — OpenCV.js 前處理 + 手動 ROI + Tesseract Worker + 多幀投票 + 領域規則

// ====== DOM ======
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const overlay = document.getElementById('ocr-overlay');
const statusMessage = document.getElementById('status-message');
const outputText = document.getElementById('output-text');
const psmSelect = document.getElementById('psm-select');
const intervalInput = document.getElementById('interval-input');
const focusThInput = document.getElementById('focus-th-input');
const clearRoiBtn = document.getElementById('clear-roi');
const container = document.getElementById('canvas-container');
const roiSelector = document.getElementById('roi-selector');
const ctx = canvas.getContext('2d');

// ====== Config ======
let OCR_INTERVAL = 400; // ms
defaultPSM = 7; // 7: line, 6: block
const WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
const CODE_REGEX = /[A-Z]{3}\d{5}[A-Z]{2}/g; // raw extraction
const ACCEPT_REGEX = /^[A-Z]{3}\d{5}[A-Z]{2}$/; // final acceptance
const CONFUSION_MAP = { '0':'O','O':'0','1':'I','I':'1','5':'S','S':'5','8':'B','B':'8','2':'Z','Z':'2','6':'G','G':'6','D':'0' };
const VALID_PREFIXES = [
  'AKE','AKH','AKN','ALF','AMA','AMJ','AMX','AAX','AAP','PAG','PGE','PLA','PMC','PMB','PZA','RKN','RAP','RSJ','RKB','RSP'
];
const VALID_AIRLINES = ['CI','BR','CX','JL','NH','CA','MU','CZ','OZ','KE','SQ','TR','QF','VA'];

// ====== State ======
let worker;            // Tesseract worker
let isRecognizing = false;
let lastOCR = 0;
let roi = null;        // {x,y,width,height} in canvas pixels
let dragStart = null;  // for ROI drawing
let recent = [];       // last N results: {code, conf, time}
const MAX_RECENT = 7;
let cvReady = false;
let streamReady = false;

// ====== Utils ======
function now(){ return performance.now(); }
function clamp(v,min,max){ return Math.max(min, Math.min(max, v)); }
function applyConfusionMap(code){ return code.split('').map(c=>CONFUSION_MAP[c]||c).join(''); }
function vote(results){
  if(results.length===0) return null;
  const score = new Map();
  for(const r of results){
    score.set(r.code, (score.get(r.code)||0) + (r.conf||0));
  }
  // sort by total confidence then recency
  return [...score.entries()].sort((a,b)=> b[1]-a[1])[0][0];
}
function isValid(code){
  if(!ACCEPT_REGEX.test(code)) return false;
  const prefix = code.slice(0,3);
  const airline = code.slice(-2);
  return VALID_PREFIXES.includes(prefix) && VALID_AIRLINES.includes(airline);
}

// Laplacian variance focus measure
function focusScore(srcCanvas){
  const mat = cv.imread(srcCanvas);
  let gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
  let lap = new cv.Mat();
  cv.Laplacian(gray, lap, cv.CV_64F);
  const mean = new cv.Mat();
  const std = new cv.Mat();
  cv.meanStdDev(lap, mean, std);
  const variance = Math.pow(std.doubleAt(0,0),2);
  mat.delete(); gray.delete(); lap.delete(); mean.delete(); std.delete();
  return variance;
}

// OpenCV preprocessing pipeline: gray -> sharpen -> adaptive thresh -> morphology close
function preprocessToCanvas(srcCanvas){
  const mat = cv.imread(srcCanvas);
  let gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);

  // unsharp mask (light sharpening)
  let blur = new cv.Mat();
  cv.GaussianBlur(gray, blur, new cv.Size(3,3), 0);
  let sharp = new cv.Mat();
  cv.addWeighted(gray, 1.5, blur, -0.5, 0, sharp);

  // adaptive threshold (Gaussian)
  let bin = new cv.Mat();
  cv.adaptiveThreshold(sharp, bin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2);

  // morphology close to connect strokes
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2,2));
  let morph = new cv.Mat();
  cv.morphologyEx(bin, morph, cv.MORPH_CLOSE, kernel);

  // render to an offscreen canvas for Tesseract
  const out = document.createElement('canvas');
  out.width = morph.cols; out.height = morph.rows;
  cv.imshow(out, morph);

  // cleanup
  mat.delete(); gray.delete(); blur.delete(); sharp.delete(); bin.delete(); kernel.delete(); morph.delete();
  return out;
}

function getDefaultROI(){
  // middle 40% height x 90% width
  const w = canvas.width, h = canvas.height;
  return { x: Math.round(w*0.05), y: Math.round(h*0.30), width: Math.round(w*0.90), height: Math.round(h*0.40) };
}

function getActiveROI(){
  return roi || getDefaultROI();
}

function drawRoiGuide(){
  // visual guide on overlay layer
  overlay.innerHTML = '';
  const r = getActiveROI();
  const box = document.createElement('div');
  box.className = 'text-box';
  box.style.position = 'absolute';
  box.style.border = '2px solid var(--accent)';
  box.style.left = r.x + 'px';
  box.style.top = r.y + 'px';
  box.style.width = r.width + 'px';
  box.style.height = r.height + 'px';
  overlay.appendChild(box);
}

// ====== ROI drag selection ======
container.addEventListener('mousedown', (e)=>{
  const rect = container.getBoundingClientRect();
  dragStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  roiSelector.hidden = false;
  roiSelector.style.left = dragStart.x + 'px';
  roiSelector.style.top = dragStart.y + 'px';
  roiSelector.style.width = '0px';
  roiSelector.style.height = '0px';
});
container.addEventListener('mousemove', (e)=>{
  if(!dragStart) return;
  const rect = container.getBoundingClientRect();
  const curr = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  const left = Math.min(dragStart.x, curr.x);
  const top = Math.min(dragStart.y, curr.y);
  const width = Math.abs(curr.x - dragStart.x);
  const height = Math.abs(curr.y - dragStart.y);
  roiSelector.style.left = left + 'px';
  roiSelector.style.top = top + 'px';
  roiSelector.style.width = width + 'px';
  roiSelector.style.height = height + 'px';
});
container.addEventListener('mouseup', ()=>{
  if(!dragStart) return;
  const rect = container.getBoundingClientRect();
  const sel = roiSelector.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  roi = {
    x: Math.round((sel.left - rect.left) * scaleX),
    y: Math.round((sel.top - rect.top) * scaleY),
    width: Math.round(sel.width * scaleX),
    height: Math.round(sel.height * scaleY),
  };
  dragStart = null;
});
clearRoiBtn.addEventListener('click', ()=>{ roi = null; roiSelector.hidden = true; });

// ====== Camera ======
async function startCamera(){
  statusMessage.textContent = '正在請求相機權限…';
  try{
    const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
    video.srcObject = stream;
    await new Promise(res=> video.onloadedmetadata = res);
    video.play();

    // set canvas size to video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    statusMessage.textContent = '相機已啟動';
    streamReady = true;
  }catch(err){
    statusMessage.textContent = '無法存取相機，請檢查權限或裝置。';
    console.error(err);
  }
}

// ====== Tesseract ======
async function initTesseract(){
  const { createWorker } = Tesseract;
  worker = await createWorker({ logger: m=>console.log(m) });
  await worker.load();
  await worker.loadLanguage('eng');
  await worker.initialize('eng');
  await worker.setParameters({
    tessedit_char_whitelist: WHITELIST,
    tessedit_pageseg_mode: String(defaultPSM), // 6 or 7
    tessedit_ocr_engine_mode: '1',            // LSTM_ONLY
  });
}

psmSelect.addEventListener('change', async ()=>{
  await worker.setParameters({ tessedit_pageseg_mode: String(psmSelect.value) });
});
intervalInput.addEventListener('change', ()=>{
  OCR_INTERVAL = clamp(Number(intervalInput.value)||400, 100, 2000);
});

// ====== Main loop ======
async function loop(){
  requestAnimationFrame(loop);
  if(!cvReady || !streamReady || isRecognizing) return;
  const t = now();
  if(t - lastOCR < OCR_INTERVAL) return;

  // draw current frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // focus gate
  const focus = focusScore(canvas);
  const focusTh = Number(focusThInput.value)||80;
  if(focus < focusTh){
    statusMessage.textContent = `等待清晰畫面…(焦點:${focus.toFixed(0)})`;
    return;
  }

  isRecognizing = true;
  try{
    // crop to ROI
    const r = getActiveROI();
    const roiCanvas = document.createElement('canvas');
    roiCanvas.width = r.width; roiCanvas.height = r.height;
    const rctx = roiCanvas.getContext('2d');
    rctx.drawImage(canvas, r.x, r.y, r.width, r.height, 0, 0, r.width, r.height);

    // preprocess
    const procCanvas = preprocessToCanvas(roiCanvas);

    const { data } = await worker.recognize(procCanvas);
    const raw = (data.text||'').replace(/\s/g,'').toUpperCase();
    const candidates = raw.match(CODE_REGEX) || [];

    let accepted = null;
    let bestConf = data.confidence || 0;

    for(let code of candidates){
      if(isValid(code)){ accepted = code; break; }
      const fixed = applyConfusionMap(code);
      if(isValid(fixed)){ accepted = fixed; break; }
    }

    if(accepted){
      recent.push({ code: accepted, conf: bestConf, time: Date.now() });
      if(recent.length > MAX_RECENT) recent.shift();
      const voted = vote(recent);
      outputText.textContent = voted || accepted;
      statusMessage.textContent = `OK (conf=${bestConf.toFixed(1)}, focus=${focus.toFixed(0)})`;
    }else{
      statusMessage.textContent = `未命中格式 (conf=${bestConf.toFixed(1)}, focus=${focus.toFixed(0)})`;
    }

    drawRoiGuide();
  }catch(err){
    console.error('OCR 異常:', err);
    statusMessage.textContent = 'OCR 失敗';
  }finally{
    lastOCR = t;
    isRecognizing = false;
  }
}

// ====== Bootstrap ======
(async function main(){
  // wait for OpenCV
  await new Promise((resolve)=>{
    if(window.cv && cv.Mat){ cvReady = true; resolve(); }
    else{
      const i = setInterval(()=>{ if(window.cv && cv.Mat){ clearInterval(i); cvReady = true; resolve(); } }, 30);
    }
  });
  await startCamera();
  await initTesseract();
  statusMessage.textContent = '就緒。拖曳框選 ROI 或直接對準櫃號。';
  loop();
})();

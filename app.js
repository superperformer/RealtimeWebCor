// 取得所有 HTML 元素
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ocrOverlay = document.getElementById('ocr-overlay');
const statusMessage = document.getElementById('status-message');
const outputText = document.getElementById('output-text');
const context = canvas.getContext('2d');

let isRecognizing = false;

// 1. 啟動相機
async function startCamera() {
    statusMessage.textContent = '正在請求相機權限...';
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            statusMessage.textContent = '相機已啟動。';
            // 設定 canvas 尺寸與影片一致
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // 啟動即時處理迴圈
            requestAnimationFrame(processFrame);
        };
    } catch (err) {
        statusMessage.textContent = '無法存取相機，請檢查權限設定。';
        console.error('無法取得相機串流:', err);
    }
}

// 2. 即時影像處理與 OCR 邏輯
async function processFrame() {
    // 只有當辨識未執行時才繼續
    if (!isRecognizing) {
        isRecognizing = true;
        
        // 將當前影片幀繪製到 canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 呼叫 OCR 辨識
        try {
            const { data: { words } } = await Tesseract.recognize(
                canvas,
                'eng+chi_tra',
                {
                    logger: m => console.log(m) // 可以在 console 觀察辨識進度
                }
            );

            // 清空舊的文字方框
            ocrOverlay.innerHTML = '';
            
            let fullText = '';
            words.forEach(word => {
                const { text, bbox } = word;
                // 將所有辨識出的文字組合成完整內容
                fullText += text + ' ';
                
                // 創建並設定文字方框元素
                const box = document.createElement('div');
                box.className = 'text-box';
                box.textContent = text;
                box.style.left = `${bbox.x0}px`;
                box.style.top = `${bbox.y0}px`;
                box.style.width = `${bbox.x1 - bbox.x0}px`;
                box.style.height = `${bbox.y1 - bbox.y0}px`;
                
                // 將方框加入到 overlay
                ocrOverlay.appendChild(box);
            });
            
            outputText.textContent = fullText;
            
        } catch (error) {
            console.error('OCR 辨識失敗:', error);
        }
        
        isRecognizing = false;
    }
    
    // 每一幀都繼續呼叫自己，建立處理迴圈
    requestAnimationFrame(processFrame);
}

// 啟動應用程式
window.addEventListener('load', startCamera);
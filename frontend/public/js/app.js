// Update with your Cloud Run URL before deploying to Firebase
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8080'
  : 'https://YOUR_CLOUD_RUN_URL';

// Replace with the actual class names from your trained model
const CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3'];

const BOX_COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899'];

const MAX_FILE_SIZE = 5 * 1024 * 1024;

const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const canvas        = document.getElementById('canvas');
const ctx           = canvas.getContext('2d');
const detectBtn     = document.getElementById('detectBtn');
const statusEl      = document.getElementById('status');
const resultsEl     = document.getElementById('results');
const fishCountEl   = document.getElementById('fishCount');
const inferenceEl   = document.getElementById('inferenceTime');
const detectionList = document.getElementById('detectionList');
const canvasContainer = document.getElementById('canvasContainer');

let currentFile  = null;
let currentImage = null;

// --- Drag and drop ---

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

// --- File handling ---

function handleFile(file) {
  if (!file) return;

  if (!file.type.startsWith('image/')) {
    showError('Please select an image file.');
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    showError('Image must be under 5MB.');
    return;
  }

  currentFile = file;

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    // Set canvas intrinsic size to image's natural size so box coordinates
    // map 1:1 — CSS handles the visual scaling to fit the container
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    canvasContainer.hidden = false;
    detectBtn.disabled = false;
    resultsEl.hidden = true;
    clearStatus();
    URL.revokeObjectURL(url);
  };
  img.onerror = () => showError('Could not load image.');
  img.src = url;
}

// --- Detection ---

detectBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  detectBtn.disabled = true;
  showStatus('Detecting…');
  resultsEl.hidden = true;

  const formData = new FormData();
  formData.append('image', currentFile);

  const start = Date.now();

  try {
    const resp = await fetch(`${API_BASE}/detect`, { method: 'POST', body: formData });
    const elapsed = Date.now() - start;

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.error || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    clearStatus();
    drawResults(data.detections);
    showResults(data.fish_count, elapsed, data.detections);
  } catch (err) {
    showError(err.message);
  } finally {
    detectBtn.disabled = false;
  }
});

// --- Drawing ---

function drawResults(detections) {
  ctx.drawImage(currentImage, 0, 0);

  const lineWidth = Math.max(2, canvas.width / 300);
  const fontSize  = Math.max(14, canvas.width / 45);
  ctx.font = `${fontSize}px sans-serif`;

  detections.forEach(det => {
    const color = BOX_COLORS[det.class_id % BOX_COLORS.length];
    const { x1, y1, x2, y2 } = det.box;
    const label = `${className(det.class_id)} ${Math.round(det.confidence * 100)}%`;

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Label pill
    const pad = 4;
    const textW = ctx.measureText(label).width;
    const pillH = fontSize + pad * 2;
    const pillY = y1 >= pillH ? y1 - pillH : y1 + (y2 - y1);
    ctx.fillStyle = color;
    ctx.fillRect(x1, pillY, textW + pad * 2, pillH);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + pad, pillY + fontSize);
  });
}

// --- Results panel ---

function showResults(count, elapsed, detections) {
  fishCountEl.textContent  = count;
  inferenceEl.textContent  = elapsed;

  detectionList.innerHTML = detections.map(det => {
    const color = BOX_COLORS[det.class_id % BOX_COLORS.length];
    return `<li>
      <span class="swatch" style="background:${color}"></span>
      ${className(det.class_id)} — ${Math.round(det.confidence * 100)}%
    </li>`;
  }).join('');

  resultsEl.hidden = false;
}

// --- Helpers ---

function className(id) {
  return CLASS_NAMES[id] ?? `Class ${id}`;
}

function showStatus(msg) {
  statusEl.textContent = msg;
  statusEl.className = 'status';
  statusEl.hidden = false;
}

function showError(msg) {
  statusEl.textContent = msg;
  statusEl.className = 'status error';
  statusEl.hidden = false;
}

function clearStatus() {
  statusEl.hidden = true;
}

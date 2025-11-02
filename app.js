// 後端 Render 網址
const API_BASE = "https://catfaceid.onrender.com";

// 影像上傳辨識
async function predict(file) {
  const formData = new FormData();
  formData.append("file", file);

  const resp = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// 取得已知標籤（可選）
async function getLabels() {
  const resp = await fetch(`${API_BASE}/labels`);
  return resp.json();
}

// 綁定 UI
document.getElementById("btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("file");
  const resultEl = document.getElementById("result");
  resultEl.textContent = "上傳中…";

  if (!fileInput.files || !fileInput.files[0]) {
    resultEl.textContent = "請先選一張貓咪照片 🐱";
    return;
  }

  try {
    const data = await predict(fileInput.files[0]);
    resultEl.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    resultEl.textContent = `辨識失敗：${e.message}`;
  }
});

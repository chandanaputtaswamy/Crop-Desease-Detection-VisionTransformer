const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const dropText = document.getElementById("dropText");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const resultsCard = document.getElementById("resultsCard");
const statusMsg = document.getElementById("statusMsg");
const apiUrlInput = document.getElementById("apiUrl");

let selectedFile = null;

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

["dragover", "dragenter"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);

dropZone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("Please select an image file.", true);
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropText.hidden = true;
  predictBtn.disabled = false;
  resultsCard.hidden = true;
  setStatus("");
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  setStatus("Analyzing image...");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
    const res = await fetch(`${baseUrl}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed with status ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    setStatus("");
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  } finally {
    predictBtn.disabled = false;
  }
});

function renderResults(data) {
  resultsCard.hidden = false;

  document.getElementById("cropTop").textContent =
    `${data.crop.top_prediction.label} (${(data.crop.top_prediction.confidence * 100).toFixed(1)}%)`;
  document.getElementById("diseaseTop").textContent =
    `${data.disease.top_prediction.label} (${(data.disease.top_prediction.confidence * 100).toFixed(1)}%)`;
  document.getElementById("partScore").textContent = data.part_score.toFixed(4);

  renderBarList("cropList", data.crop.top_k);
  renderBarList("diseaseList", data.disease.top_k);
}

function renderBarList(elementId, items) {
  const list = document.getElementById(elementId);
  list.innerHTML = "";
  items.forEach((item) => {
    const pct = (item.confidence * 100).toFixed(1);
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="bar-row"><span>${item.label}</span><span>${pct}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
    `;
    list.appendChild(li);
  });
}

function setStatus(msg, isError = false) {
  statusMsg.textContent = msg;
  statusMsg.classList.toggle("error", isError);
}

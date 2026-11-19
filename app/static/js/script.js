// =========================================================
// Mango Fruit Disease Detection - Frontend Controller
// With Device Camera Capture & Proportion-Preserving Auto-Resizing
// =========================================================

document.addEventListener("DOMContentLoaded", () => {
  // Tabs & Mode Switching
  const tabUploadBtn = document.getElementById("tabUploadBtn");
  const tabCameraBtn = document.getElementById("tabCameraBtn");
  const uploadPanel = document.getElementById("uploadPanel");
  const cameraPanel = document.getElementById("cameraPanel");
  let activeTab = "upload"; // "upload" or "camera"

  // Upload Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");

  // Camera Elements
  const cameraStartBox = document.getElementById("cameraStartBox");
  const startCameraBtn = document.getElementById("startCameraBtn");
  const cameraStreamWrapper = document.getElementById("cameraStreamWrapper");
  const cameraVideo = document.getElementById("cameraVideo");
  const capturePhotoBtn = document.getElementById("capturePhotoBtn");
  const switchCameraBtn = document.getElementById("switchCameraBtn");
  const closeCameraBtn = document.getElementById("closeCameraBtn");
  const cameraCanvas = document.getElementById("cameraCanvas");
  let mediaStream = null;
  let currentFacingMode = "environment"; // Prefer rear camera on mobile

  // Preview Elements
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const previewFilename = document.getElementById("previewFilename");
  const previewFilesize = document.getElementById("previewFilesize");
  const retakeChangeBtn = document.getElementById("retakeChangeBtn");

  // Detection & Loading Elements
  const detectBtn = document.getElementById("detectBtn");
  const loadingBox = document.getElementById("loadingBox");
  const resultCardWrapper = document.getElementById("resultCardWrapper");
  const alertBox = document.getElementById("alertBox");
  const alertText = document.getElementById("alertText");
  const uploadAnotherBtn = document.getElementById("uploadAnotherBtn");

  // Result Elements
  const resultHeader = document.getElementById("resultHeader");
  const resultDiseaseTitle = document.getElementById("resultDiseaseTitle");
  const resultScientificName = document.getElementById("resultScientificName");
  const resultBadge = document.getElementById("resultBadge");
  const resultImg = document.getElementById("resultImg");
  const resultConfidence = document.getElementById("resultConfidence");
  const resultConfidenceBar = document.getElementById("resultConfidenceBar");
  const resultDescription = document.getElementById("resultDescription");
  const resultSymptoms = document.getElementById("resultSymptoms");
  const resultManagement = document.getElementById("resultManagement");
  const resOriginalDim = document.getElementById("resOriginalDim");
  const probsToggleBtn = document.getElementById("probsToggleBtn");
  const probsList = document.getElementById("probsList");

  let currentFile = null;

  // -------------------------------------------------------
  // Input Options: Tabs Switching
  // -------------------------------------------------------
  tabUploadBtn.addEventListener("click", () => {
    if (activeTab === "upload") return;
    activeTab = "upload";
    tabUploadBtn.classList.add("active");
    tabCameraBtn.classList.remove("active");
    uploadPanel.classList.add("active");
    cameraPanel.classList.remove("active");
    stopCamera();
    hideAlert();
  });

  tabCameraBtn.addEventListener("click", () => {
    if (activeTab === "camera") return;
    activeTab = "camera";
    tabCameraBtn.classList.add("active");
    tabUploadBtn.classList.remove("active");
    cameraPanel.classList.add("active");
    uploadPanel.classList.remove("active");
    hideAlert();
  });

  // -------------------------------------------------------
  // Camera Functionality (getUserMedia)
  // -------------------------------------------------------
  startCameraBtn.addEventListener("click", startCamera);

  async function startCamera() {
    hideAlert();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showAlert("Camera access is not supported by your browser or environment. Please use the 'Upload from Device' option.");
      return;
    }

    try {
      // Request camera with preference for rear camera on mobile
      const constraints = {
        video: {
          facingMode: { ideal: currentFacingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };

      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraVideo.srcObject = mediaStream;
      await cameraVideo.play();

      cameraStartBox.style.display = "none";
      cameraStreamWrapper.style.display = "block";
    } catch (err) {
      console.error("Camera Error:", err);
      let errorMsg = "Unable to access camera.";

      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        errorMsg = "Camera permission was denied. Please allow camera permissions in your browser or use 'Upload from Device'.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        errorMsg = "No camera found on this device. Please use 'Upload from Device'.";
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        errorMsg = "Camera is already in use by another application.";
      }

      showAlert(errorMsg);
      stopCamera();
    }
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    cameraVideo.srcObject = null;
    cameraStreamWrapper.style.display = "none";
    cameraStartBox.style.display = "flex";
  }

  closeCameraBtn.addEventListener("click", stopCamera);

  // Switch / Flip Camera (Toggle between environment and user)
  switchCameraBtn.addEventListener("click", async () => {
    currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
    stopCamera();
    await startCamera();
  });

  // Capture Photo from Camera
  capturePhotoBtn.addEventListener("click", () => {
    if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
      showAlert("Camera stream not ready. Please wait a moment.");
      return;
    }

    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    const ctx = cameraCanvas.getContext("2d");
    ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

    cameraCanvas.toBlob(
      (blob) => {
        if (!blob) {
          showAlert("Failed to capture snapshot from camera.");
          return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const filename = `camera_mango_${timestamp}.jpg`;
        const file = new File([blob], filename, { type: "image/jpeg" });

        // Stop camera stream immediately to conserve battery and privacy
        stopCamera();

        // Process file for preview
        handleFileSelection(file, cameraCanvas.toDataURL("image/jpeg", 0.95));
      },
      "image/jpeg",
      0.95
    );
  });

  // -------------------------------------------------------
  // Drag and Drop & Browse File Input
  // -------------------------------------------------------
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  dropZone.addEventListener("click", (e) => {
    if (e.target !== browseBtn) {
      fileInput.click();
    }
  });

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  // -------------------------------------------------------
  // Handle Selected / Captured Image File
  // -------------------------------------------------------
  function handleFileSelection(file, directDataUrl = null) {
    hideAlert();
    resultCardWrapper.style.display = "none";

    // Validate type
    const validExtensions = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
    const extension = file.name.split(".").pop().toLowerCase();
    const isValidExt = ["jpg", "jpeg", "png", "webp"].includes(extension);

    if (!validExtensions.includes(file.type) && !isValidExt) {
      showAlert("Invalid file format. Please provide a JPG, JPEG, or PNG image.");
      return;
    }

    // Validate size (max 16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
      showAlert("File is too large. Maximum allowed size is 16 MB.");
      return;
    }

    currentFile = file;

    // Display metadata
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    function showPreview(dataUrl) {
      previewImg.src = dataUrl;
      uploadPanel.style.display = "none";
      cameraPanel.style.display = "none";
      previewContainer.style.display = "block";
      detectBtn.disabled = false;
    }

    if (directDataUrl) {
      showPreview(directDataUrl);
    } else {
      const reader = new FileReader();
      reader.onload = (e) => showPreview(e.target.result);
      reader.onerror = () => showAlert("Failed to read image. Please try again.");
      reader.readAsDataURL(file);
    }
  }

  // -------------------------------------------------------
  // Change / Retake Photo Button
  // -------------------------------------------------------
  function resetSelection() {
    currentFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewContainer.style.display = "none";
    detectBtn.disabled = true;
    hideAlert();

    // Restore active panel
    if (activeTab === "upload") {
      uploadPanel.style.display = "block";
      cameraPanel.style.display = "none";
    } else {
      cameraPanel.style.display = "block";
      uploadPanel.style.display = "none";
      stopCamera();
    }
  }

  retakeChangeBtn.addEventListener("click", resetSelection);

  // -------------------------------------------------------
  // Detect Disease (Submit to Flask Backend)
  // -------------------------------------------------------
  detectBtn.addEventListener("click", async () => {
    if (!currentFile) {
      showAlert("Please select or capture a mango fruit photograph first.");
      return;
    }

    // UI Loading State
    detectBtn.disabled = true;
    loadingBox.style.display = "block";
    resultCardWrapper.style.display = "none";
    hideAlert();

    const formData = new FormData();
    formData.append("image", currentFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      // Verify server responded with JSON before parsing
      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        const errText = await response.text();
        console.error("Non-JSON Server Response:", errText);
        if (response.status === 502) {
          throw new Error("Server temporarily ran out of memory processing this image (502 Bad Gateway). Please try again or use a smaller image.");
        } else if (response.status === 504) {
          throw new Error("Server took too long to respond (504 Gateway Timeout). Please try again.");
        } else {
          throw new Error(`Server returned HTTP ${response.status} (${response.statusText}). Check server logs for details.`);
        }
      }

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "An error occurred while analyzing the image.");
      }

      // Display prediction result
      renderResult(data);

    } catch (err) {
      console.error("Prediction Error:", err);
      showAlert(err.message || "Failed to communicate with prediction server.");
    } finally {
      loadingBox.style.display = "none";
      detectBtn.disabled = false;
    }
  });

  // -------------------------------------------------------
  // Render Prediction Result
  // -------------------------------------------------------
  function renderResult(data) {
    const isHealthy = data.is_healthy;
    const info = data.info || {};

    // Header Styling
    resultHeader.className = isHealthy ? "result-header healthy" : "result-header diseased";

    // Titles & Badge
    resultDiseaseTitle.textContent = data.display_name;
    resultScientificName.textContent = info.scientific_name || "";
    
    if (isHealthy) {
      resultBadge.className = "result-badge badge-healthy";
      resultBadge.innerHTML = "✓ Healthy Condition";
    } else {
      resultBadge.className = "result-badge badge-diseased";
      resultBadge.innerHTML = `⚠️ Disease Detected (${info.severity || "Active"} Severity)`;
    }

    // Image, Resolution & Confidence
    resultImg.src = data.image_url;
    resOriginalDim.textContent = data.original_resolution || "Uploaded Image";

    resultConfidence.textContent = `${data.confidence}%`;
    resultConfidenceBar.style.width = "0%";
    setTimeout(() => {
      resultConfidenceBar.style.width = `${Math.min(data.confidence, 100)}%`;
    }, 100);

    // Overview Description
    resultDescription.textContent = info.description || "No description provided.";

    // Symptoms List
    resultSymptoms.innerHTML = "";
    if (info.symptoms && info.symptoms.length > 0) {
      info.symptoms.forEach((symptom) => {
        const li = document.createElement("li");
        li.textContent = symptom;
        resultSymptoms.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "No specific symptoms reported.";
      resultSymptoms.appendChild(li);
    }

    // Management Tips List
    resultManagement.innerHTML = "";
    if (info.management && info.management.length > 0) {
      info.management.forEach((tip) => {
        const li = document.createElement("li");
        li.textContent = tip;
        resultManagement.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "Maintain regular orchard care.";
      resultManagement.appendChild(li);
    }

    // Probability breakdown table
    renderProbabilities(data.top_probabilities || []);

    // Show result card and scroll smoothly
    resultCardWrapper.style.display = "block";
    resultCardWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // -------------------------------------------------------
  // Probability Breakdown Accordion
  // -------------------------------------------------------
  function renderProbabilities(probabilities) {
    probsList.innerHTML = "";
    probabilities.forEach((p) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span style="font-weight: 500;">${p.class_name}</span>
        <span style="font-weight: 700; color: ${p.confidence > 50 ? 'var(--mango-green-dark)' : 'var(--text-muted)'};">${p.confidence}%</span>
      `;
      probsList.appendChild(row);
    });
  }

  probsToggleBtn.addEventListener("click", () => {
    const isVisible = probsList.style.display === "block";
    probsList.style.display = isVisible ? "none" : "block";
    probsToggleBtn.innerHTML = isVisible
      ? "<span>▼</span> Show All Class Probabilities"
      : "<span>▲</span> Hide Class Probabilities";
  });

  // -------------------------------------------------------
  // Test Another Mango Image Action
  // -------------------------------------------------------
  uploadAnotherBtn.addEventListener("click", () => {
    resetSelection();
    resultCardWrapper.style.display = "none";
    document.getElementById("detectionSection").scrollIntoView({ behavior: "smooth" });
  });

  // -------------------------------------------------------
  // Utilities
  // -------------------------------------------------------
  function showAlert(msg) {
    alertText.textContent = msg;
    alertBox.style.display = "flex";
  }

  function hideAlert() {
    alertBox.style.display = "none";
    alertText.textContent = "";
  }

  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  // Stop camera when user navigates away or closes tab
  window.addEventListener("beforeunload", stopCamera);
});

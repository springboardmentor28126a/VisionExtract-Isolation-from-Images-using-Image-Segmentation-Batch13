document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const loader = document.getElementById("loader");
    const resultsContainer = document.getElementById("resultsContainer");
    const originalPreview = document.getElementById("originalPreview");
    const resultPreview = document.getElementById("resultPreview");
    const downloadBtn = document.getElementById("downloadBtn");

    // Click to upload
    dropZone.addEventListener("click", () => fileInput.click());

    // Drag and Drop Events
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file.");
            return;
        }

        // Preview original image immediately
        originalPreview.src = URL.createObjectURL(file);
        
        // Update UI states
        resultsContainer.style.display = "none";
        loader.style.display = "block";

        // Send to FastAPI Backend
        extractSubject(file);
    }

    async function extractSubject(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            // Note: Make sure your api.py is running on port 8000!
            const response = await fetch("http://127.0.0.1:8000/extract", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            // The backend returns an image blob
            const resultBlob = await response.blob();
            const resultUrl = URL.createObjectURL(resultBlob);

            // Display the result and configure the download button
            resultPreview.src = resultUrl;
            downloadBtn.href = resultUrl;

            // Update UI states
            loader.style.display = "none";
            resultsContainer.style.display = "grid";

        } catch (error) {
            console.error("Inference Error:", error);
            alert("Failed to process image. Ensure the backend API is running on port 8000.");
            loader.style.display = "none";
        }
    }
});

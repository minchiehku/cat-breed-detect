async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const resultDiv = document.getElementById("result");
    const uploadedImage = document.getElementById("uploadedImage");

    if (!fileInput.files[0]) {
        resultDiv.textContent = "Please select an image.";
        return;
    }

    const originalFile = fileInput.files[0];
    console.log(`Original file size: ${(originalFile.size / 1024 / 1024).toFixed(2)} MB`);

    // 如果圖片超過 5MB，壓縮處理
    let fileToUpload = originalFile;
    if (originalFile.size > 5 * 1024 * 1024) {
        resultDiv.textContent = "Compressing image...";
        fileToUpload = await compressImage(originalFile, 1280, 1280); // 壓縮到最大邊長 1280px
        console.log(`Compressed file size: ${(fileToUpload.size / 1024 / 1024).toFixed(2)} MB`);
    }

    // 顯示壓縮後的圖片預覽
    const reader = new FileReader();
    reader.onload = function (e) {
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = "block";
    };
    reader.readAsDataURL(fileToUpload);

    // 上傳圖片
    const formData = new FormData();
    formData.append("file", fileToUpload);

    try {
        resultDiv.textContent = "Classifying...";
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        resultDiv.textContent = `Predicted breed: ${data.breed_name}, Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    } catch (error) {
        resultDiv.textContent = `Error: ${error.message}`;
    }
}

// 圖片壓縮函數
async function compressImage(file, maxWidth, maxHeight) {
    return new Promise((resolve) => {
        const img = new Image();
        const reader = new FileReader();

        reader.onload = function (e) {
            img.src = e.target.result;
        };

        img.onload = function () {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");

            let width = img.width;
            let height = img.height;

            // 調整尺寸比例
            if (width > maxWidth || height > maxHeight) {
                if (width > height) {
                    height *= maxWidth / width;
                    width = maxWidth;
                } else {
                    width *= maxHeight / height;
                    height = maxHeight;
                }
            }

            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);

            // 壓縮圖片，質量設置為 0.8
            canvas.toBlob(
                (blob) => {
                    resolve(new File([blob], file.name, { type: file.type }));
                },
                file.type,
                0.8
            );
        };

        reader.readAsDataURL(file);
    });
}

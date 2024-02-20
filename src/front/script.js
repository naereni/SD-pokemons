const textInput = document.getElementById("textInput");
const getImageBtn = document.getElementById("getImageBtn");
const generationText = document.getElementById("generationText");
const imageContainer = document.getElementById("imageContainer");

getImageBtn.addEventListener("click", async () => {
    try {
        const text = textInput.value.trim();
        if (text === "") {
            throw new Error("Text input is empty");
        }

        // Показываем текст "Генерация..."
        generationText.style.display = "block";

        const response = await fetch("/generate-stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error("Failed to fetch image");
        }

        // Прячем текст "Генерация..." после получения ответа
        generationText.style.display = "none";

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        const image = document.createElement("img");
        image.src = imageUrl;
        imageContainer.innerHTML = "";
        imageContainer.appendChild(image);
    } catch (error) {
        console.error(error);
    }
});

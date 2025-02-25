document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const progressBarContainer = document.querySelector('.progress-bar-container');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent'); // Get percentage element
    const analyzingText = document.getElementById('analyzingText'); // Get analyzing text element
    const resultSection = document.querySelector('.result-section');

    imageUpload.addEventListener('change', () => {
        imagePreview.innerHTML = ''; // Clear previous preview
        resultSection.style.display = 'none'; // Hide result section on new upload
        progressBarContainer.style.display = 'none'; // Hide progress bar initially
        analyzingText.style.display = 'none'; // Hide analyzing text initially

        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('uploaded-img'); // Add class for styling
                imagePreview.appendChild(img);
            }
            reader.readAsDataURL(file);
        }
    });

    const classifyButton = document.querySelector('.classify-button');
    if (classifyButton) {
        classifyButton.addEventListener('click', () => {
            progressBarContainer.style.display = 'block';
            analyzingText.style.display = 'block'; // Show analyzing text
            progressBar.style.width = '0%'; // Reset progress to 0
            progressPercent.textContent = '0%'; // Reset percentage text

            // Simulate progress (replace with actual progress updates if needed)
            let progress = 0;
            const interval = setInterval(() => {
                progress += 10; // Increment progress (adjust speed as needed)
                progressBar.style.width = progress + '%';
                progressPercent.textContent = progress + '%'; // Update percentage text
                if (progress >= 100) {
                    clearInterval(interval);
                    // In a real application, you'd get the final result from the backend here
                    // and update the result section.
                }
            }, 200); // Update every 200ms (adjust timing as needed)
        });
    }
});
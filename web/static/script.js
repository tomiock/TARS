document.addEventListener("DOMContentLoaded", function () {
    const storedData = localStorage.getItem("searchFormData");
    const filters = storedData ? JSON.parse(storedData) : null;

    if (!filters) return;

    const formData = new FormData();
    for (const key in filters) {
        formData.append(key, filters[key]);
    }

    fetch("/recommend", {
        method: "POST",
        body: formData
    })

    .then(response => response.json())
    .then(data => {
        console.log(data); // <-- Aquí ves la estructura real

        const translatorsList = document.getElementById("translators-list");
        data.translators.forEach(translator => {
            const card = document.createElement("div");
            card.classList.add("translator-card");

            card.innerHTML = `
                    <div class="translator-details">
                        <h3>${translator.TRANSLATOR}</h3>
                        <p><strong>Quality:</strong> ${translator.QUALITY_EVALUATION_mean.toFixed(1)}/10</p>
                        <p><strong>Hourly Rate:</strong> $${translator.HOURLY_RATE_mean.toFixed(2)}</p>
                        <p><strong>Industry:</strong> ${translator.MANUFACTURER_INDUSTRY}</p>
                        <p><strong>Source Lang:</strong> ${translator.SOURCE_LANG}</p>
                        <p><strong>Target Lang:</strong> ${translator.TARGET_LANG}</p>
                    </div>
                    <div class="translator-buttons-and-cost">
                        <button class="accept-button">Accept</button>
                        <button class="info-button">More Info</button>
                    </div>
                `;
            translatorsList.appendChild(card);

            const infoButton = card.querySelector(".info-button");
            infoButton.addEventListener("click", function () {
                const panel = document.getElementById("translator-info-panel");
                const translatorInfoText = document.getElementById("translator-details-text");
                translatorInfoText.textContent = `Details about ${translator.name}: ${translator.details || "No additional details available."}`;
                panel.classList.add("show");
            });
        });
    })
    .catch(error => console.error("Error loading translators:", error));
        // Traductores alternativos simulados (normalmente esto vendría del backend)
    const alternativeTranslators = [
    { name: "Lucía García", languages: "ES→EN", quality: 7, available: false },
    { name: "Marco Rossi", languages: "IT→FR", quality: 8, available: true },
    { name: "Lena Müller", languages: "DE→EN", quality: 6, available: true },
    ];

    const altList = document.getElementById("alt-list");
    const lowerQualityCheckbox = document.getElementById("allow-lower-quality");
    const partialAvailabilityCheckbox = document.getElementById("allow-partial-availability");

    function updateAlternativeSuggestions() {
    altList.innerHTML = "";

    const allowLowQ = lowerQualityCheckbox.checked;
    const allowPartial = partialAvailabilityCheckbox.checked;

    const filtered = alternativeTranslators.filter(t => {
        const qualityOK = allowLowQ ? t.quality >= 7 : t.quality >= 8;
        const availabilityOK = allowPartial ? true : t.available;
        return qualityOK && availabilityOK;
    });

    if (filtered.length === 0) {
        altList.innerHTML = "<li>No suggestions available with current settings.</li>";
    } else {
        filtered.forEach(t => {
        const li = document.createElement("li");
        li.textContent = `${t.name} (${t.languages}) – Quality: ${t.quality}`;
        altList.appendChild(li);
        });
    }
    }

    lowerQualityCheckbox.addEventListener("change", updateAlternativeSuggestions);
    partialAvailabilityCheckbox.addEventListener("change", updateAlternativeSuggestions);

    // Inicialización
    updateAlternativeSuggestions();

});


document.addEventListener("DOMContentLoaded", function () {
    // Botón de búsqueda de traductores
    const searchButton = document.getElementById("search-translators");

    if (searchButton) {
        searchButton.addEventListener("click", function (event) {
            event.preventDefault(); // Evita el envío normal

            const formData = new FormData(document.getElementById("formulario"));
            const formObject = {};
            formData.forEach((value, key) => {
                formObject[key] = value;
            });

            // Guardamos en localStorage
            localStorage.setItem("searchFormData", JSON.stringify(formObject));

            // Redirigimos
            window.location.href = "translators_avail.html";
        });
    }

    // Botón "Back to Form"
    const backButton = document.getElementById("back-to-form");
    if (backButton) {
        backButton.addEventListener("click", function () {
            window.location.href = "forms.html";
        });
    }

    // Cierre del panel lateral
    const closePanel = document.getElementById("close-panel");
    if (closePanel) {
        closePanel.addEventListener("click", function () {
            const panel = document.getElementById("translator-info-panel");
            panel.classList.remove("show");
        });
    }

    // Rellenar el formulario con los filtros almacenados (si aplica)
    const storedData = localStorage.getItem("searchFormData");
    const filters = storedData ? JSON.parse(storedData) : null;
    if (filters) {
        for (const key in filters) {
            const input = document.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = filters[key];
            }
        }
    }
});

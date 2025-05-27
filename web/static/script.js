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
        const translatorsList = document.getElementById("translators-list");
        data.translators.forEach(translator => {
            const card = document.createElement("div");
            card.classList.add("translator-card");

            card.innerHTML = `
                <img src="${translator.avatar}" alt="Avatar de ${translator.name}" class="avatar">
                <div class="translator-details">        
                    <h3>${translator.name}</h3>
                    <p><strong>Quality:</strong> ${translator.quality}/10</p>
                    <p><strong>Available Hours:</strong> ${translator.hours_available}</p>
                    <p><strong>Finish Date:</strong> ${translator.finish_day}</p>
                    <p><strong>Completion Time:</strong> ${translator.done_in}</p>
                </div>
                <div class="translator-buttons-and-cost">
                    <p class="translator-cost"><strong>Cost:</strong> ${translator.cost}</p>
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

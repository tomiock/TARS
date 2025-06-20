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
        if (!translatorsList) {
            console.error("Element with ID 'translators-list' not found.");
            return;
        }

        data.translators.forEach(translator => {
            const card = document.createElement("div");
            card.classList.add("translator-card");

            card.innerHTML = `
                <div class="translator-details">
                    <h3>${translator.TRANSLATOR}</h3>
                    <p><strong>Quality:</strong> ${translator.QUALITY_EVALUATION_mean.toFixed(1)}/10</p>
                    <p><strong>Avg Hourly Rate:</strong> $${translator.HOURLY_RATE_mean.toFixed(2)}</p>
                    <p><strong>Primary Industry:</strong> ${translator.MANUFACTURER_INDUSTRY}</p>
                    <p><strong>Best Source Lang:</strong> ${translator.SOURCE_LANG}</p>
                    <p><strong>Best Target Lang:</strong> ${translator.TARGET_LANG}</p>
                </div>
                <div class="translator-buttons-and-cost">
                    <button class="accept-button">Accept</button>
                    <button class="info-button">More Info</button>
                </div>
            `;
            translatorsList.appendChild(card);

            const infoButton = card.querySelector(".info-button");
            if (infoButton) {
                infoButton.addEventListener("click", function () {
                    const panel = document.getElementById("translator-info-panel");
                    const translatorInfoText = document.getElementById("translator-details-text");

                    if (!panel || !translatorInfoText) {
                        console.error("Info panel elements not found.");
                        return;
                    }

                    let detailsHTML = `
                        <h2>${translator.TRANSLATOR}</h2>
                        <p><strong>Quality Evaluation:</strong> ${translator.QUALITY_EVALUATION_mean.toFixed(1)}/10</p>
                        <p><strong>Average Hourly Rate:</strong> $${translator.HOURLY_RATE_mean.toFixed(2)}</p>
                        <p><strong>Primary Industry:</strong> ${translator.MANUFACTURER_INDUSTRY}</p>
                        <p><strong>Source Language:</strong> ${translator.SOURCE_LANG}</p>
                        <p><strong>Target Language:</strong> ${translator.TARGET_LANG}</p>
                    `;

                    if (translator.details) {
                        detailsHTML += `<p><strong>Additional Details:</strong> ${translator.details}</p>`;
                    } else {
                        detailsHTML += `<p><strong>Additional Details:</strong> No additional details provided.</p>`;
                    }

                    translatorInfoText.innerHTML = detailsHTML;
                    panel.classList.add("show");
                });
            }

            // MODIFICATION FOR ACCEPT BUTTON POPUP STARTS HERE
            const acceptButton = card.querySelector(".accept-button");
            if (acceptButton) {
                acceptButton.addEventListener("click", function() {
                    // Remove any existing popup first to avoid stacking
                    const existingPopup = document.getElementById("assignment-popup");
                    if (existingPopup) {
                        existingPopup.remove();
                    }

                    const popup = document.createElement("div");
                    popup.id = "assignment-popup"; // Added an ID for easier removal
                    popup.textContent = "Translator Assigned";

                    // Styling the popup
                    popup.style.position = "fixed";
                    popup.style.bottom = "200px";
                    popup.style.left = "50%";
                    popup.style.transform = "translateX(-50%)";
                    popup.style.backgroundColor = "#d4edda"; // A light green color
                    popup.style.color = "#155724"; // A dark green text color for contrast
                    popup.style.padding = "12px 25px";
                    popup.style.borderRadius = "8px";
                    popup.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.1)";
                    popup.style.zIndex = "1000";
                    popup.style.fontSize = "22px";
                    popup.style.opacity = "0"; // Start transparent for fade-in effect
                    popup.style.transition = "opacity 0.3s ease-in-out";


                    document.body.appendChild(popup);

                    // Trigger reflow to enable transition
                    setTimeout(() => {
                        popup.style.opacity = "1";
                    }, 10);


                    // Remove the popup after a few seconds
                    setTimeout(() => {
                        popup.style.opacity = "0";
                        // Wait for fade-out transition to complete before removing from DOM
                        setTimeout(() => {
                            if (popup.parentNode) {
                                popup.remove();
                            }
                        }, 300); // This duration should match the transition duration
                    }, 1500); // Popup visible for 3 seconds
                });
            }
            // MODIFICATION FOR ACCEPT BUTTON POPUP ENDS HERE
        });
    })
    .catch(error => console.error("Error loading translators:", error));

    const altList = document.getElementById("alt-list");
    const lowerQualityCheckbox = document.getElementById("allow-lower-quality");
    const partialAvailabilityCheckbox = document.getElementById("allow-partial-availability");

    function updateAlternativeSuggestions() {
        if (!altList || !lowerQualityCheckbox || !partialAvailabilityCheckbox) return; // Guard clause

        altList.innerHTML = "";

        const allowLowQ = lowerQualityCheckbox.checked;
        const allowPartial = partialAvailabilityCheckbox.checked;

        // Traductores alternativos simulados (normalmente esto vendría del backend)
        const alternativeTranslators = [
            { name: "Lucía García", languages: "ES→EN", quality: 7, available: false },
            { name: "Marco Rossi", languages: "IT→FR", quality: 8, available: true },
            { name: "Lena Müller", languages: "DE→EN", quality: 6, available: true },
        ];

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

    if (lowerQualityCheckbox && partialAvailabilityCheckbox && altList) {
        lowerQualityCheckbox.addEventListener("change", updateAlternativeSuggestions);
        partialAvailabilityCheckbox.addEventListener("change", updateAlternativeSuggestions);
        updateAlternativeSuggestions(); // Initial call
    }
});


document.addEventListener("DOMContentLoaded", function () {
    const searchButton = document.getElementById("search-translators");
    if (searchButton) {
        searchButton.addEventListener("click", function (event) {
            event.preventDefault();
            const form = document.getElementById("formulario");
            if (form) {
                const formData = new FormData(form);
                const formObject = {};
                formData.forEach((value, key) => {
                    formObject[key] = value;
                });
                localStorage.setItem("searchFormData", JSON.stringify(formObject));
                window.location.href = "translators_avail.html";
            }
        });
    }

    const backButton = document.getElementById("back-to-form");
    if (backButton) {
        backButton.addEventListener("click", function () {
            window.location.href = "forms.html";
        });
    }

    const closePanelButton = document.getElementById("close-panel");
    if (closePanelButton) {
        closePanelButton.addEventListener("click", function () {
            const panel = document.getElementById("translator-info-panel");
            if (panel) {
                panel.classList.remove("show");
            }
        });
    }

    const storedDataOnFormPage = localStorage.getItem("searchFormData");
    const filtersOnFormPage = storedDataOnFormPage ? JSON.parse(storedDataOnFormPage) : null;
    if (filtersOnFormPage) {
        for (const key in filtersOnFormPage) {
            const input = document.querySelector(`[name="${key}"]`);
            if (input) {
                if (input.type === "checkbox" || input.type === "radio") {
                    input.checked = filtersOnFormPage[key] === input.value || filtersOnFormPage[key] === true;
                } else {
                    input.value = filtersOnFormPage[key];
                }
            }
        }
    }
});

document.addEventListener("DOMContentLoaded", function () {
    let translatorsData = []; // Variable para almacenar los traductores

    // Cargar los traductores desde el JSON
    fetch("Traductores.json")
        .then(response => response.json())
        .then(data => {
            translatorsData = data.translators; // Guardamos los traductores
            displayTranslators(translatorsData); // Mostramos todos al inicio
        })
        .catch(error => console.error("Error al cargar los datos de traductores:", error));

    function displayTranslators(translators) {
        const container = document.getElementById("translators-list");
        container.innerHTML = ""; // Limpiar antes de agregar nuevos datos

        if (translators.length === 0) {
            container.innerHTML = "<p>No se encontraron traductores.</p>";
            return;
        }

        translators.forEach(translator => {
            const translatorDiv = document.createElement("div");
            translatorDiv.classList.add("translator-card");
            translatorDiv.innerHTML = `
                <img src="${translator.avatar}" alt="Avatar de ${translator.name}" class="avatar">
                <h3>${translator.name}</h3>
                <p><strong>Idiomas:</strong> ${translator.languages.join(", ")}</p>
                <p><strong>Disponibilidad:</strong> ${translator.availability}</p>
                <p><strong>Calidad:</strong> ${translator.quality}/10</p>
                <h4>Historial de Proyectos:</h4>
                <ul>
                    ${translator.history.map(proj => `<li>${proj.project} (${proj.languagePair})</li>`).join("")}
                </ul>
            `;
            container.appendChild(translatorDiv);
        });
    }

    // Evento para filtrar por calidad mínima
    document.getElementById("filter-quality").addEventListener("click", function () {
        const minQuality = parseInt(document.getElementById("min-quality").value, 10);
        if (!isNaN(minQuality)) {
            const filteredTranslators = translatorsData.filter(translator => translator.quality >= minQuality);
            displayTranslators(filteredTranslators);
        }
    });
});

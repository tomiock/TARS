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
            const percentage = (translator.quality / 10) * 180; // Convertimos calidad a grados (0-180)
            const offset = 180 - percentage; // Ajuste correcto de la barra semicircular
        
            const translatorDiv = document.createElement("div");
            translatorDiv.classList.add("translator-card");
            translatorDiv.innerHTML = `
                <img src="${translator.avatar}" alt="Avatar de ${translator.name}" class="avatar">
                <h3>${translator.name}</h3>
                <p><strong>Idiomas:</strong> ${translator.languages.join(", ")}</p>
                <p><strong>Disponibilidad:</strong> ${translator.availability}</p>
        
                <!-- Barra semicircular de calidad -->
                <div class="quality-container">
                    <svg viewBox="0 0 100 50" class="quality-meter">
                        <path d="M 10 50 A 40 40 0 0 1 90 50" stroke="#ddd" stroke-width="10" fill="transparent"/>
                        <path d="M 10 50 A 40 40 0 0 1 90 50" stroke="#28a745" stroke-width="10" fill="transparent"
                            stroke-dasharray="180" stroke-dashoffset="${offset}"/>
                    </svg>
                    <p class="quality-text">${translator.quality}/10</p>
                </div>
            
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


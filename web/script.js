

document.addEventListener("DOMContentLoaded", function() {      //procesamos la información del json
    fetch("diezTraductores.json")       
        .then(response => response.json())
        .then(data => {
            const translatorsList = document.getElementById("translators-list");
            data.translators.forEach(translator => {
                const card = document.createElement("div");
                card.classList.add("translator-card");      //creamos cada una de las tarjetas de los traductores con su información
                                                            //dividimos las tarjetas en tres secciones: avatar, detalles (nombre y horarios) y el coste y los botones
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

                // Añadir el event listener para el botón de "Más Información"
                const infoButton = card.querySelector(".info-button");
                infoButton.addEventListener("click", function () {
                    // Mostramos el panel de información
                    const panel = document.getElementById("translator-info-panel");
                    const translatorInfoText = document.getElementById("translator-details-text");

                    // Aquí puedes poner la información adicional del traductor
                    translatorInfoText.textContent = `Details about ${translator.name}: ${translator.details || "No additional details available."}`;
                    
                    panel.classList.add("show");
                });
            });

        })
        .catch(error => console.error("Error loading translators:", error));            //por si aca
});

document.addEventListener("DOMContentLoaded", function () {             //al pulsar el botón, nos manda a la página de los traductores
    const searchButton = document.getElementById("search-translators");

    if (searchButton) {
        searchButton.addEventListener("click", function (event) {
            event.preventDefault(); // Evita que el formulario se envíe normalmente
            window.location.href = "translators_avail.html"; 
        });
    }
});

document.getElementById("back-to-form").addEventListener("click", function() {      //botón provisional para cambiar de página
    window.location.href = "forms.html"; 
});


document.getElementById("close-panel").addEventListener("click", function () {
    const panel = document.getElementById("translator-info-panel");
    panel.classList.remove("show");
});
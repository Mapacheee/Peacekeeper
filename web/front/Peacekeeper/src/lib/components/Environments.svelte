<section class="environments">
    <h2>Protección en todo tipo de entornos</h2>
    <div class="cards">
        <div class="card">
            <img src="enviroments/viapublica.png" alt="Vía pública" />
            <p>Vía pública</p>
        </div>
        <div class="card">
            <img src="enviroments/salaclase.png" alt="Sala de clases" />
            <p>Sala de clases</p>
        </div>
        <div class="card">
            <img src="enviroments/parque.png" alt="Parques" />
            <p>Parques</p>
        </div>
        <div class="card">
            <img src="enviroments/cancha.png" alt="Cancha" />
            <p>Cancha</p>
        </div>
    </div>
</section>

<style>
    h2 {
        color: #000000;
        font-family: 'DM Sans', sans-serif;

    }
    .environments {

        text-align: center;
        padding: 2rem;
    }
    .cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .card {
        text-align: center;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff;
    }
</style>

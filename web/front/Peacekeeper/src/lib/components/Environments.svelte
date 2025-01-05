<section class="environments">
    <h2>Protección en todo tipo de entornos</h2>
    <div class="cards">
        <div class="card">
            <!-- <img src="path/to/image1.jpg" alt="Vía pública" /> -->
            <p>Vía pública</p>
        </div>
        <div class="card">
            <!-- <img src="path/to/image2.jpg" alt="Sala de clases" /> -->
            <p>Sala de clases</p>
        </div>
        <div class="card">
            <!-- <img src="path/to/image3.jpg" alt="Parques" /> -->
            <p>Parques</p>
        </div>
        <div class="card">
            <!-- <img src="path/to/image4.jpg" alt="Cancha" /> -->
            <p>Cancha</p>
        </div>
    </div>
</section>

<style>
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

        /* img {
            width: 100%;
            height: auto;
            border-radius: 5px;
        } */
    }
</style>

<section class="hero">
    <div class="content">
        <img class="camera-image" src="camera.png" alt="2 cameras" />
        <div class="text-content">
            <h1>Protección en tiempo real con tecnología avanzada.</h1>
            <p>
                Detecta y previene situaciones de violencia con nuestro sistema
                inteligente de análisis de imágenes.
            </p>
            <button class="cta-button">Saber más</button>
        </div>
    </div>
</section>

<style>
    .hero {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        background: linear-gradient(to bottom, #eeeeee, #999999);
    }

    .content {
        display: flex;
        align-items: center;
        gap: 2rem;
    }

    .camera-image {
        margin-left: -3%;
        max-width: 70%;
        transform: scaleX(-1);
    }

    .text-content {
        text-align: left;
    }

    .text-content h1 {
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }

    .text-content p {
        margin-bottom: 1rem;
        font-size: 1rem;
        color: #555;
    }

    .cta-button {
        background-color: #333;
        color: #fff;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1rem;
    }

    .cta-button:hover {
        background-color: #555;
    }
</style>

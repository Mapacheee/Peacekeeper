<script lang="ts">
    import MediaQuery from 'svelte-media-queries'
</script>

<section class="hero">
    <img class="camera-image" src="camera.png" alt="2 cameras" />
    <div class="text-content">
        <h1>Protección en tiempo real con tecnología avanzada.</h1>
        <p>
            Detecta y previene situaciones de violencia con nuestro sistema
            inteligente de análisis de imágenes.
        </p>
        <button class="cta-button">Saber más</button>
    </div>
</section>

<style>
    .hero {
        display: flex;
        justify-content: center;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 2rem;
        width: 100%;
        background: linear-gradient(to bottom, #eeeeee, #999999);
    }

    @media (max-width: 530px) {
        .hero {
            flex-wrap: wrap;
        }

        div.text-content {
            width: 80%;
            text-align: center;
            margin: 0 auto;
            padding: 0;
        }
    }

    .camera-image {
        max-width: 70%;
        transform: scaleX(-1);
    }

    .text-content {
        text-align: left;
        padding-right: 5rem;
        padding-left: 2rem;
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
    }

    .cta-button:hover {
        background-color: #555;
    }
</style>

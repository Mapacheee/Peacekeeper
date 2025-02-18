<script lang="ts">
    function handleScrollToPlans() {
        const plans = document.querySelector('#plans')
        plans?.scrollIntoView({ behavior: 'smooth' })
    }
    function handleScrollToEnvironments() {
        const information = document.querySelector('#environments')
        information?.scrollIntoView({ behavior: 'smooth' })
    }
</script>

<section class="hero">
    <div class="content">
        <img class="camera-image" src="camera.png" alt="cameras" />
        <div class="text-content">
            <h1>Protección en tiempo real con tecnología avanzada.</h1>
            <p>Detecta y previene situaciones de violencia con</p>
            <p>nuestro sistema inteligente de análisis de imágenes.</p>
            <button onclick={handleScrollToPlans}>Ver planes</button>
            <button
                onclick={handleScrollToEnvironments}
                class="secondary outline">Saber más</button
            >
        </div>
    </div>
</section>

<style>
    :root {
        --primary-button-bg-color: #0f1620;
    }
    .hero {
        margin-top: -2%;
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
        margin-left: -23%;
        max-width: 70%;
        transform: scaleX(-1);
    }

    .text-content {
        margin-left: 15%;
        text-align: center;
    }

    .text-content h1 {
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: -0.05rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        color: #000000;
    }

    .text-content p {
        line-height: 0.4rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 200;
        margin-bottom: 1rem;
        font-size: 1rem;
        color: #000000;
    }

    button {
        font-family: 'Agdasima', sans-serif;
        background-color: var(--primary-button-bg-color);
        color: #bdbdbd;
        border: none;
        text-decoration: none;
        padding: 0.3rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1.2rem;
        transition: background-color 0.3s ease;

        &:hover {
            background-color: #555;
        }

        &.secondary {
            background-color: transparent;
            color: #000000;
            outline: 1px solid var(--primary-button-bg-color);
        }
    }
</style>

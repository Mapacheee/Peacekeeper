<script lang="ts">
    import '../../global.css'
    import '@picocss/pico'
</script>

<div class="dark-theme" data-theme="dark">
    <a href="/">
        <img
            src="/logo/pk_only_logo.png"
            class="brand-logo"
            alt="peacekeeper logo"
        />
    </a>

    <slot />
</div>

<style>
    .dark-theme {
        background-color: var(--pico-background-color);
        color: var(--pico-color);

        min-height: 100vh;
    }
    .brand-logo {
        position: absolute;
        width: 80px;
        height: 80px;
        margin: 1rem;
        background-color: #c2c7d0;
        border-radius: 50%;
    }
</style>

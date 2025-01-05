<script lang="ts">
    import './global.css'
    import '@picocss/pico'

    import Navbar from '$lib/components/Navbar.svelte'
    import Footer from '$lib/components/Footer.svelte'
</script>

<Navbar />
<div class="main-content">
    <slot />
</div>
<Footer />

<style>
    .main-content {
        min-height: calc(
            100vh - 100px /* Subtracting approximate navbar height */
        );
    }
</style>

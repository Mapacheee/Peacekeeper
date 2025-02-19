<script lang="ts">
    import '../global.css'
    import '@picocss/pico'

    import Navbar from '$lib/components/Navbar.svelte'
    import AboutFooter from '$lib/components/AboutFooter.svelte'
</script>

<Navbar />
<div class="main-content">
    <slot />
</div>
<AboutFooter />

<style>
    .main-content {
        min-height: calc(
            100vh - 100px /* Subtracting approximate navbar height */
        );
    }
</style>

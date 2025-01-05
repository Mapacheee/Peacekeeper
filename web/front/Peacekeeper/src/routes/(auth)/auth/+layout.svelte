<script lang="ts">
    import '../../global.css'
    import '@picocss/pico'
</script>

<a href="/">
    <img
        src="/logo/pk_only_logo.png"
        class="brand-logo"
        alt="peacekeeper logo"
    />
</a>

<slot />

<style>
    .brand-logo {
        width: 80px;
        height: 80px;
    }
</style>

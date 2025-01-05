<script>
    import { page } from '$app/stores'

    $: currentHash = $page.url.hash || '#home'
</script>

<nav class="header">
    <img src="/logo/pk_logo.png" alt="a Peacekeeper" class="logo" />
    <ul class="menu">
        <li>
            <a href="#home" class={currentHash === '#home' ? 'active' : ''}
                >Inicio</a
            >
        </li>
        <li>
            <a
                href="#services"
                class={currentHash === '#services' ? 'active' : ''}>Servicio</a
            >
        </li>
        <li>
            <a href="#plans" class={currentHash === '#plans' ? 'active' : ''}
                >Planes</a
            >
        </li>
        <li>
            <a href="#about" class={currentHash === '#about' ? 'active' : ''}
                >Sobre nosotros</a
            >
        </li>
    </ul>
    <a href="https://panel.peacekeeper.com" class="panel-text" target="_blank"
        >Panel</a
    >
</nav>

<style>
    .menu a {
        font-family: 'Agdasima';
        font-size: 150%;
        font-weight: 700;
        color: #000000;
        text-decoration: none;
        position: relative;
        transition: color 0.3s ease;
    }

    .menu a:hover,
    .menu a:focus {
        color: #555555;
    }

    .menu a::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: -5px;
        width: 0;
        height: 2px;
        background-color: #555555;
        transition: width 0.3s ease;
    }

    .menu a:hover::after,
    .menu a:focus::after {
        width: 100%;
    }

    .menu a.active {
        color: #000000;
    }

    .menu a.active::after {
        width: 100%;
        background-color: #555555;
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #eeeeee;
    }

    .logo {
        margin-left: -7%;
        height: 100px;
        width: auto;
        object-fit: contain;
    }

    .menu {
        margin-left: -16%;
        display: flex;
        gap: 3rem;
    }

    .menu li {
        list-style: none;
    }

    .panel-text {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        transition: color 0.3s ease;
        margin-right: 1%;
        color: white;
        background-color: #0f1620;
        border-radius: 11%;
        font-family: 'Agdasima', sans-serif;
        font-size: 150%;
        font-weight: 700;
        height: auto;
        width: 8%;
        cursor: pointer;
        text-decoration: none;
    }

    .panel-text:hover {
        color: #555555;
    }
</style>

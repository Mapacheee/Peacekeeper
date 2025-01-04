<nav class="header">
    <img src="/logo/pk_logo.png" alt="a Peacekeeper" class="logo" />
    <ul class="menu">
        <li><a href="#home">Inicio</a></li>
        <li><a href="#services">Servicio</a></li>
        <li><a href="#plans">Planes</a></li>
        <li><a href="#about">Sobre nosotros</a></li>
    </ul>
    <a href="https://panel.peacekeeper.com" class="panel-text" target="_blank"
        >Panel</a
    >
</nav>

<style>
    .menu {
        margin-left: -16%;
        display: flex;
        gap: 3rem;

        a {
            font-family: 'Agdasima';
            font-size: 200%;
            font-weight: 700;
            color: #000000;
            transition: color 0.3s ease;
            text-decoration: none;
            color: #000000;

            &:hover {
                color: #555555;
            }
        }
        li {
            list-style: none;
        }
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #eeeeee;
        /*border-bottom: 1px solid #ddd;*/
    }
    .logo {
        margin-left: -7%;
        height: 100px;
        width: auto;
        object-fit: contain;
    }

    .panel-text {
        transition: color 0.3s ease;
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
        font-size: 200%;
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

<script lang="ts">
    import { page } from '$app/stores'
    import MediaQuery from 'svelte-media-queries'
    // @ts-ignore
    import MenuIcon from 'svelte-icons/ti/TiThMenu.svelte'

    $: pathSegments = $page.url.pathname.split('/')
    $: pathName = pathSegments[pathSegments.length - 1] || '/'
    let isMenuOpen = false
    let isMovileSize = true

    const toggleMenu = () => (isMenuOpen = !isMenuOpen)
</script>

<MediaQuery query="(max-width: 850px)" bind:matches={isMovileSize} />

<nav class={isMovileSize ? 'mobile-navbar' : 'navbar'}>
    {#if isMovileSize}
        <ul>
            <li>
                <button class="menu-button" on:click={toggleMenu}>
                    <MenuIcon />
                </button>
            </li>
        </ul>
        <ul>
            <li>
                <a href="/" class="brand-logo">
                    <img src="/logo/pk_only_logo.png" alt="a Peacekeeper" />
                    PEACEKEEPER
                </a>
            </li>
        </ul>
        <ul>
            <li></li>
        </ul>
    {:else}
        <ul>
            <li>
                <a href="/" class="brand-logo">
                    <img src="/logo/pk_only_logo.png" alt="a Peacekeeper" />
                    PEACEKEEPER
                </a>
            </li>
        </ul>

        <ul class="desktop-menu">
            {#each [['/', 'Inicio'], ['services', 'Servicio'], ['plans', 'Planes']] as [hash, text]}
                <li>
                    <a href={hash} class={pathName === hash ? 'active' : ''}>
                        {text}
                    </a>
                </li>
            {/each}
            <li>
                <a href="auth" class="panel-text">Sign in</a>
            </li>
        </ul>
    {/if}
</nav>
{#if isMovileSize && isMenuOpen}
    <aside>
        {#each [['/', 'Inicio'], ['services', 'Servicio'], ['plans', 'Planes']] as [hash, text]}
            <li>
                <a
                    href={hash}
                    class={pathName === hash ? 'active' : ''}
                    on:click={toggleMenu}
                >
                    {text}
                </a>
            </li>
        {/each}
        <li>
            <a href="auth" class="panel-text">Sign in</a>
        </li>
    </aside>
{/if}

<style>
    .navbar {
        background-color: #eeeeee;
        padding-right: 1rem;
    }

    .mobile-navbar {
        background-color: #eeeeee;
    }

    .menu-button {
        width: 4rem;
        height: 4rem;
        color: #000000;
        background-color: transparent;
        border: none;
        &:focus {
            box-shadow: none;
        }
    }

    .brand-logo {
        display: flex;
        align-items: center;
        text-decoration: none;
        font-size: 2rem;
        letter-spacing: 0.1em;
        color: #000000;
        font-weight: bold;

        img {
            height: 90px;
        }

        &:hover {
            color: #000000;
        }
    }

    .desktop-menu {
        gap: 3rem;

        a {
            font-size: 150%;
            color: #000000;
            text-decoration: none;
            position: relative;
            transition: color 0.3s ease;

            &:hover,
            &:focus {
                color: #555555;
                &::after {
                    width: 100%;
                }
            }

            &::after {
                content: '';
                position: absolute;
                left: 0;
                bottom: -5px;
                width: 0;
                height: 2px;
                background-color: #555555;
                transition: width 0.3s ease;
            }

            &.active {
                color: #000000;
                &::after {
                    width: 100%;
                    background-color: #555555;
                }
            }
        }
    }

    a.panel-text {
        background-color: #0f1620;
        color: #eeeeee !important;
        padding: 0.5rem 1rem;
        position: unset;
        font-size: 120%;
        /* border-radius: 11%; */

        transition: color 0.3s ease;

        &:hover {
            background-color: transparent;
            color: #000000 !important;
            transition:
                background-color 0.3s ease,
                color 0.3s ease;
        }
    }

    aside {
        a {
            text-decoration: none;
        }
    }
</style>

<script lang="ts">
    import MediaQuery from 'svelte-media-queries'
    // @ts-ignore
    import MenuIcon from 'svelte-icons/ti/TiThMenu.svelte'
    import BrandLogo from './BrandLogo.svelte'

    let isMovileSize: boolean
    let isMenuOpen = false

    function handleToggleMenu() {
        isMenuOpen = !isMenuOpen
    }

    function handleScrollToElement(elementId: string) {
        const plans = document.querySelector(`#${elementId}`)
        plans?.scrollIntoView({ behavior: 'smooth' })
    }

    const pages: Array<Array<string>> = [
        ['Inicio', ''],
        ['Planes', 'plans'],
        ['About', 'about']
    ]
    // TODO: Improve all navbar styles
</script>

<MediaQuery query="(max-width: 850px)" bind:matches={isMovileSize} />

<nav class="navbar">
    {#if isMovileSize}
        <button class="mobile-menu-button" onclick={handleToggleMenu}>
            <MenuIcon />
        </button>
        <div class="brand-logo">
            <BrandLogo />
        </div>
    {:else}
        <ul>
            <li>
                <BrandLogo />
            </li>
        </ul>

        <ul class="desktop-menu">
            {#each pages as [name, id]}
                <li>
                    <button
                        class={name == 'Inicio' ? 'active' : ''}
                        onclick={() => handleScrollToElement(id)}
                    >
                        {name}
                    </button>
                </li>
            {/each}
            <li>
                <a href="auth" class="button">Login</a>
            </li>
        </ul>
    {/if}
    {#if isMovileSize && isMenuOpen}
        <aside>
            {#each pages as [name, id]}
                <li>
                    <button
                        role="link"
                        class={name == 'Inicio' ? 'active' : ''}
                        onclick={() => handleScrollToElement(id)}
                    >
                        {name}
                    </button>
                </li>
            {/each}
            <li>
                <a href="auth" class="button">Login</a>
            </li>
        </aside>
    {/if}
</nav>

<style>
    .navbar {
        background-color: #eeeeee;
        padding-right: 2rem;
    }

    @media (max-width: 850px) {
        .navbar {
            background-color: #eeeeee;
            padding: 0 1rem;
        }
    }

    li {
        text-align: center;
        line-height: 1.3rem;
    }

    .brand-logo {
        width: 100%;
        display: flex;
        justify-content: center;
    }

    .mobile-menu-button {
        width: 4rem;
        height: 4rem;
        color: #000000;
        background-color: transparent;
        border: none;

        &:focus {
            box-shadow: none;
        }
    }

    .desktop-menu {
        gap: 3rem;

        button {
            font-size: 150%;
            color: #000000;
            text-decoration: none;
            position: relative;
            transition: color 0.3s ease;
            background-color: transparent;
            border-color: transparent;
            outline: none;
            border: none;

            &:hover,
            &:focus {
                color: #555555;
                box-shadow: none;
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

    a.button {
        background-color: #0f1620;
        color: #eeeeee !important;
        padding: 0.3rem 1rem;
        position: unset;
        font-size: 120%;
        border-radius: 10px;
        transition: color 0.3s ease;

        &:hover {
            background-color: transparent;
            color: #000000 !important;
            text-decoration: none;
            transition:
                background-color 0.3s ease,
                color 0.3s ease;
        }
    }

    aside {
        button {
            text-decoration: none;
            border: none;
        }
    }
</style>

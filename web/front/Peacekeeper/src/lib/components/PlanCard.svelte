<script lang="ts">
    type TextUrl = [string, string]
    export type Plan = {
        name: string
        price: number
        description: string
        features: Array<string | TextUrl>
        type: 'mensual' | 'anual' | 'único'
    }
    type Props = { plan: Plan }

    let { plan }: Props = $props()
</script>

<article>
    <h3>{plan.name} ({plan.type})</h3>
    <h5>
        clp ${plan.price.toLocaleString('es-CL')}/{plan.type === 'mensual'
            ? 'mes'
            : plan.type}
    </h5>
    <p>{plan.description}</p>
    <p>Este plan incluye:</p>
    <ul>
        {#each plan.features as feature}
            <li>
                {#if typeof feature === 'string'}
                    {feature}
                {:else}
                    <a href={feature[1]}>{feature[0]}</a>
                {/if}
            </li>
        {/each}
    </ul>
    <footer>
        <a role="button" href=".">Elegir plan</a>
    </footer>
</article>

<style>
    article {
        footer {
            text-align: center;
            a {
                width: 100%;
            }
        }
    }

    a[role='button'] {
        font-family: 'Agdasima', sans-serif;
        background-color: #0f1620;
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
    }
</style>

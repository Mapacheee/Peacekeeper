<script lang="ts">
    import PlanCard from './PlanCard.svelte'
    import type { Plan } from './PlanCard.svelte'

    const userPlans: Array<Plan> = [
        {
            name: 'Pack de inicio',
            price: 190_000,
            description:
                'Pago único para 1 mes de uso, para nuevos usuarios que buscan probar nuestro servicio.',
            features: [
                ['Cámara inteligente', '.'],
                'Monitoreo de 1 cámara',
                '1 usuario',
                '1 semana de almacenamiento'
            ],
            type: 'único'
        },
        {
            name: 'Plan básico',
            price: 100_000,
            description:
                'Diseñado para usuarios que buscan una solución básica para monitorear su hogar.',
            features: [
                'Monitoreo de 1 cámara',
                '1 usuario',
                '2 semanas de almacenamiento'
            ],
            type: 'mensual'
        },
        {
            name: 'Vía pública',
            price: 250_000,
            description:
                'Este plan es perfecto para una comunidad que busca proteger sus calles.',
            features: [
                'Monitoreo de 8 cámaras',
                '1 administrador',
                '1 semana de almacenamiento'
            ],
            type: 'anual'
        }
    ]
    const enterprisePlans: Array<Plan> = [
        {
            name: 'Plan empresarial',
            price: 500_000,
            description:
                'Diseñado para empresas que buscan proteger sus instalaciones.',
            features: [
                'Monitoreo de 10 cámaras',
                '2 administradores',
                '1 mes de almacenamiento'
            ],
            type: 'mensual'
        },
        {
            name: 'Plan corporativo',
            price: 1_000_000,
            description:
                'Este plan es perfecto para empresas que buscan proteger sus sucursales.',
            features: [
                'Monitoreo de 20 cámaras',
                '3 administradores',
                '2 meses de almacenamiento'
            ],
            type: 'mensual'
        }
    ]
</script>

<section class="environments">
    <h2>Planes para usuarios</h2>
    <div class="grid">
        {#each userPlans as plan}
            <PlanCard {plan} />
        {/each}
    </div>
    <h2>Planes para empresas</h2>
    <div class="grid">
        {#each enterprisePlans as plan}
            <PlanCard {plan} />
        {/each}
    </div>
</section>

<style>
    h2 {
        color: #000000;
        font-family: 'DM Sans', sans-serif;
    }
    section > h2 {
        text-align: center;
    }
    .grid {
        padding: 2rem;
    }
</style>

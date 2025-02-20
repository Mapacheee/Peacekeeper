<script lang="ts">
    import './global.css'
    import ValidatedInput from '$lib/components/ValidatedInput.svelte'

    import {
        emailRules,
        passwordRules
    } from '$lib/utils/form-validation-rules.js'

    let email: string | null = $state(null)
    let password: string | null = $state(null)

    function handleSubmit(event: Event) {
        event.preventDefault()
        console.log({ email, password })
    }
</script>

<main class="container form-container">
    <article>
        <h4>Inicio de sesión</h4>
        <form method="post" onsubmit={handleSubmit} novalidate>
            <fieldset>
                <ValidatedInput
                    label="Email"
                    type="email"
                    name="email"
                    placeholder="jane@gmail.com"
                    autocomplete="email"
                    validationRules={emailRules}
                    bind:value={email}
                />
                <ValidatedInput
                    label="Password"
                    type="password"
                    name="password"
                    placeholder="password"
                    autocomplete="current-password"
                    validationRules={passwordRules}
                    bind:value={password}
                />
                <small>
                    <a href="/auth/forgot-password">¿Olvidaste tu contraseña?</a
                    >
                </small>
            </fieldset>

            <input type="submit" value="Inicia sesión" />
            <small>
                ¿Aún no eres un miembro?
                <a href="/auth/register">Registrate aquí</a>
            </small>
        </form>
    </article>
</main>

<style>
</style>

<script lang="ts">
    import ValidatedInput from '$lib/components/ValidatedInput.svelte'
    import {
        emailRules,
        passwordRules
    } from '$lib/utils/form-validation-rules.js'
    import { tryLogin } from '$lib/services/auth-fetch.js'

    let email: string | null = $state(null)
    let password: string | null = $state(null)
    let isLoading: boolean = $state(false)
    let inputErrorMsg: string | null = $state(null)

    async function handleSubmit(event: Event) {
        event.preventDefault()
        if (email === null || password === null) return
        isLoading = true
        try {
            const response = await tryLogin({ email, password })
            console.log('Login successful:', response)
            // TODO: Handle successful login (e.g., store token, redirect)
            inputErrorMsg = null
            window.location.href = '/dashboard'
        } catch (err) {
            inputErrorMsg = 'Email o contraseña incorrectos'
            console.error('Login error:', err)
            // TODO: Handle login error (e.g., show error message)
        } finally {
            isLoading = false
        }
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

            <button type="submit">
                {#if isLoading}
                    <span aria-busy="true"></span>
                {:else}
                    Inicia sesión
                {/if}
            </button>

            {#if inputErrorMsg}
                <small class="error">
                    {inputErrorMsg}
                </small>
            {/if}
            <small>
                ¿Aún no eres un miembro?
                <a href="/auth/register">Registrate aquí</a>
            </small>
        </form>
    </article>
</main>

<style>
</style>

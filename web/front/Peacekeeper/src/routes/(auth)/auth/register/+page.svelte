<script lang="ts">
    import ValidatedInput from '$lib/components/ValidatedInput.svelte'
    import type { ValidationRule } from '$lib/components/ValidatedInput.svelte'
    import { tryRegister } from '$lib/services/auth-fetch.js'
    import {
        emailRules,
        passwordRules
    } from '$lib/utils/form-validation-rules.js'

    let email: string | null = $state(null)
    let password: string | null = $state(null)
    let confirmPassword: string | null = $state(null)
    let names: string | null = $state(null)
    let lastnames: string | null = $state(null)
    let isLoading: boolean = $state(false)
    let inputErrorMsg: string | null = $state(null)

    let confirmPasswordRules: Array<ValidationRule> = [
        {
            test: (value: string) => password === value && password !== null,
            message: 'las contraseñas deben ser iguales'
        }
    ]

    async function handleSubmit(event: Event) {
        event.preventDefault()
        if (
            email === null ||
            password === null ||
            lastnames === null ||
            names === null
        )
            return

        isLoading = true
        try {
            const response = await tryRegister({
                email,
                password,
                lastnames,
                names
            })
            console.log('Login successful:', response)
            // TODO: Handle successful register (e.g., store token, redirect)
            inputErrorMsg = null
            window.location.href = '/dashboard'
        } catch (err) {
            inputErrorMsg = 'Porfavor, verifica tus datos'
            console.error('Login error:', err)
            // TODO: Handle register error (e.g., show error message)
        } finally {
            isLoading = false
        }
    }
</script>

<main class="container form-container">
    <article>
        <h4>Registrarse</h4>
        <form method="post" onsubmit={handleSubmit} novalidate>
            <fieldset class="grid">
                <ValidatedInput
                    label="Nombres"
                    type="text"
                    name="name"
                    placeholder="Jane"
                    autocomplete="given-name"
                    validationRules={[]}
                    bind:value={names}
                />
                <ValidatedInput
                    label="Apellidos"
                    type="text"
                    name="lastname"
                    placeholder="Doe"
                    autocomplete="family-name"
                    validationRules={[]}
                    bind:value={lastnames}
                />
            </fieldset>
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
            </fieldset>
            <fieldset class="grid">
                <ValidatedInput
                    label="Contraseña"
                    type="password"
                    name="password"
                    placeholder="contraseña"
                    autocomplete="new-password"
                    validationRules={[...passwordRules]}
                    bind:value={password}
                />
                <ValidatedInput
                    label="Confirmar contraseña"
                    type="password"
                    name="confirm-password"
                    placeholder="contraseña"
                    autocomplete="new-password"
                    validationRules={confirmPasswordRules}
                    bind:value={confirmPassword}
                />
            </fieldset>

            <button type="submit">
                {#if isLoading}
                    <span aria-busy="true"></span>
                {:else}
                    Registrarse
                {/if}
            </button>
            {#if inputErrorMsg}
                <small class="error">
                    {inputErrorMsg}
                </small>
            {/if}

            <small>
                ¿Ya estás registrado?
                <a href="/auth/">Inicia sesión</a>
            </small>
        </form>
    </article>
</main>

<style>
</style>

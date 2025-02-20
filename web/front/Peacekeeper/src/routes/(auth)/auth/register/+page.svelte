<script lang="ts">
    import '../global.css'
    import ValidatedInput from '$lib/components/ValidatedInput.svelte'
    import type { ValidationRule } from '$lib/components/ValidatedInput.svelte'
    import {
        emailRules,
        passwordRules
    } from '$lib/utils/form-validation-rules.js'

    let email: string | null = $state(null)
    let password: string | null = $state(null)
    let confirmPassword: string | null = $state(null)
    let name: string | null = $state(null)
    let lastname: string | null = $state(null)

    let confirmPasswordRules: Array<ValidationRule> = [
        {
            test: (value: string) => password === value && password !== null,
            message: 'las contraseñas deben ser iguales'
        }
    ]

    function handleSubmit(event: Event) {
        event.preventDefault()
        console.log({ email, password, name, lastname, confirmPassword })
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
                    bind:value={name}
                />
                <ValidatedInput
                    label="Apellidos"
                    type="text"
                    name="lastname"
                    placeholder="Doe"
                    autocomplete="family-name"
                    validationRules={[]}
                    bind:value={lastname}
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

            <input type="submit" value="Registrarse" />
            <small>
                ¿Ya estás registrado?
                <a href="/auth/">Inicia sesión</a>
            </small>
        </form>
    </article>
</main>

<style>
</style>

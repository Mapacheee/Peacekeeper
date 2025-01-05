<script>
    import './global.css'

    // TODO: add form validation
</script>

<main class="container form-container">
    <form method="post">
        <fieldset>
            <label>
                Email
                <input
                    type="email"
                    name="email"
                    placeholder="jane@gmail.com"
                    autocomplete="email"
                />
                <small id="email-helper" class="error">
                    you must provide a valid email address
                </small>
            </label>
            <label>
                Contraseña
                <input
                    type="password"
                    name="password"
                    placeholder="contraseña"
                    autocomplete="current-password"
                />
                <small id="password-helper" class="error hide">
                    you must provide a password
                </small>
            </label>
        </fieldset>

        <input type="submit" value="Inicia sección" />
        <small>
            ¿Aún no eres un miembro?
            <a href="/auth/register">Registrate aquí</a>
        </small>
    </form>
</main>

<style></style>

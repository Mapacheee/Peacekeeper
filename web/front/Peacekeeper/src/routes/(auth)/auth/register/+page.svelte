<script>
    import '../global.css'

    // TODO: add form validation
</script>

<main class="container form-container">
    <form method="post">
        <fieldset class="grid">
            <label>
                Nombres
                <input
                    type="text"
                    name="name"
                    autocomplete="given-name"
                    placeholder="Jane"
                />
            </label>
            <label>
                Apellidos
                <input
                    type="text"
                    name="lastname"
                    autocomplete="family-name"
                    placeholder="Doe"
                />
            </label>
        </fieldset>
        <fieldset>
            <label>
                Email
                <input
                    type="email"
                    name="email"
                    placeholder="janne@gmail.com"
                    autocomplete="email"
                />
                <small id="email-helper" class="error">
                    you must provide a valid email address
                </small>
            </label>
        </fieldset>
        <fieldset class="grid">
            <label>
                Contraseña
                <input
                    type="password"
                    name="password"
                    autocomplete="new-password"
                    placeholder="contraseña"
                />

                <small id="password-helper" class="error hide">
                    you must provide a password
                </small>
            </label>
            <label>
                Confirma contraseña
                <input
                    type="password"
                    name="confirm-password"
                    autocomplete="new-password"
                    placeholder="contraseña"
                />
                <small id="password-helper" class="error hide">
                    you must provide a password
                </small>
            </label>
        </fieldset>

        <input type="submit" value="Registrarse" />
        <small>
            ¿Ya estás registrado?
            <a href="/auth/">Inicia sección</a>
        </small>
    </form>
</main>

<style>
</style>

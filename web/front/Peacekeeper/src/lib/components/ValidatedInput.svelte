<script lang="ts">
    import type { HTMLInputTypeAttribute } from 'svelte/elements'

    export type ValidationRule = {
        test: (value: string) => boolean
        message: string
    }
    type Props = {
        label: string
        type: HTMLInputTypeAttribute
        name: string
        placeholder: string
        value: string | null
        rules: ValidationRule[]
        autocomplete: AutoFill | null
    }

    let {
        label,
        type,
        name,
        placeholder,
        value = $bindable(),
        rules,
        autocomplete
    }: Props = $props()
    let errorMsg: string = $state('')

    function validate(inputValue: string): string | null {
        if (!inputValue) {
            errorMsg = 'Este campo no puede estar vacío'
            return null
        }

        for (const rule of rules) {
            if (!rule.test(inputValue)) {
                errorMsg = rule.message
                return null
            }
        }

        errorMsg = ''
        return inputValue
    }

    function handleInput(event: Event) {
        const target = event.target as HTMLInputElement
        value = validate(target.value)
    }
</script>

<label>
    {label}
    <input
        {type}
        {name}
        {placeholder}
        {value}
        {autocomplete}
        oninput={handleInput}
    />
    {#if errorMsg}
        <small class="error">{errorMsg}</small>
    {/if}
</label>

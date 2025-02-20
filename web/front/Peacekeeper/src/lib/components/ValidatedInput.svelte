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
        validationRules: ValidationRule[]
        autocomplete: AutoFill | null
        isOptional?: boolean
    }

    let {
        label,
        type,
        name,
        placeholder,
        value = $bindable(),
        validationRules,
        autocomplete,
        isOptional = false
    }: Props = $props()
    let errorMsg: string = $state('')

    function validate(inputValue: string): string | null {
        if (!inputValue && !isOptional) {
            errorMsg = 'Este campo no puede estar vacío'
            return null
        }

        for (const rule of validationRules) {
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
        onchange={handleInput}
    />
    {#if errorMsg}
        <small class="error">{errorMsg}</small>
    {/if}
</label>

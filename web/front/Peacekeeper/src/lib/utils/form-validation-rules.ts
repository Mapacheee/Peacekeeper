import type { ValidationRule } from '../components/ValidatedInput.svelte'

export const emailRules: Array<ValidationRule> = [
    {
        test: (value: string) => value.includes('@'),
        message: "el email debe contener el símbolo '@'"
    },
    {
        test: (value: string) =>
            /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$/.test(value),
        message: 'formato no válido (e.j. usuario@dominio.com)'
    }
]
export const passwordRules: Array<ValidationRule> = [
    {
        test: (value: string) => value.length >= 8,
        message: 'la contraseña debe tener al menos 8 caracteres'
    },
    {
        test: (value: string) => /[a-z]/.test(value),
        message: 'la contraseña debe tener al menos una letra minúscula'
    },
    {
        test: (value: string) => /[A-Z]/.test(value),
        message: 'la contraseña debe tener al menos una letra mayúscula'
    }
]

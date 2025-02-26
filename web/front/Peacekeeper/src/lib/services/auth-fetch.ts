import axios from 'axios'
import type {
    LoginCredentials,
    LoginResponse,
    RegisterCredentials
} from '../database/auth/login.ts'

export async function tryLogin(
    credentials: LoginCredentials
): Promise<LoginResponse> | never {
    try {
        const response = await axios.post('/api/auth/login', credentials)
        return response.data
    } catch (error) {
        throw error
    }
}

export async function tryRegister(
    credentials: RegisterCredentials
): Promise<RegisterCredentials> | never {
    try {
        const response = await axios.post('/api/auth/register', credentials)
        return response.data
    } catch (error) {
        throw error
    }
}

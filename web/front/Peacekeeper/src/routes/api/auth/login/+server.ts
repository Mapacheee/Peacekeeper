import type {
    ErrorResponse,
    LoginCredentials,
    LoginResponse
} from '$lib/database/auth/login.js'
import type { RequestEvent } from '@sveltejs/kit'
import { json } from '@sveltejs/kit'

export async function POST({ request }: RequestEvent): Promise<Response> {
    try {
        const data: unknown = await request.json()
        const { email, password } = data as LoginCredentials // TODO: Add a verification step for the request values

        // TODO: Replace this with actual database validation
        if (email === 'test@example.com' && password === 'Password123') {
            return json({
                user: 'Test User',
                token: 'mock-jwt-token'
            } satisfies LoginResponse)
        }

        return json({ error: 'Invalid credentials' } satisfies ErrorResponse, {
            status: 401
        })
    } catch (error) {
        return json({ error: 'Server error' } satisfies ErrorResponse, {
            status: 500
        })
    }
}

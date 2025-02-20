import type {
    ErrorResponse,
    RegisterCredentials,
    RegisterResponse
} from '$lib/database/auth/login.js'
import type { RequestEvent } from '@sveltejs/kit'
import { json } from '@sveltejs/kit'

// Mock database for testing
const mockUsers: RegisterCredentials[] = []

export async function POST({ request }: RequestEvent): Promise<Response> {
    try {
        const data: unknown = await request.json()
        const { email, password, lastnames, names } =
            data as RegisterCredentials // TODO: Add a verification step for the request values

        if (!email || !password || !lastnames || !names) {
            return json(
                { error: 'All fields are required' } satisfies ErrorResponse,
                { status: 400 }
            )
        }

        if (mockUsers.some(user => user.email === email)) {
            return json(
                { error: 'Email already registered' } satisfies ErrorResponse,
                { status: 409 }
            )
        }

        const newUser = {
            email,
            password, // TODO: Implement jwt
            lastnames,
            names
        }

        mockUsers.push(newUser)

        return json({
            user: names,
            token: 'mock-jwt-token'
        } satisfies RegisterResponse)
    } catch (error) {
        console.error('Registration error:', error)
        return json({ error: 'Server error' } satisfies ErrorResponse, {
            status: 500
        })
    }
}

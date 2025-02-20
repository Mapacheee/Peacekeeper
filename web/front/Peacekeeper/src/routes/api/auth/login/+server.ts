import { json } from '@sveltejs/kit'
import type { RequestEvent } from '@sveltejs/kit'

export async function POST({ request }: RequestEvent) {
    try {
        const { email, password } = await request.json()

        // TODO: Replace this with actual database validation
        if (email === 'test@example.com' && password === 'Password123') {
            return json({
                user: 'Test User',
                token: 'mock-jwt-token'
            })
        }

        return json({ error: 'Invalid credentials' }, { status: 401 })
    } catch (error) {
        return json({ error: 'Server error' }, { status: 500 })
    }
}

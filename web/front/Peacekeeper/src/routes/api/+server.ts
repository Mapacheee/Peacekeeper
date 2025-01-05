import { json } from '@sveltejs/kit'

export function GET() {
    return json({
        user: 'pedrito',
        email: 'aa@gmail.com',
        password: '1234'
    })
}

export type LoginCredentials = {
    email: string
    password: string
}

export type LoginResponse = {
    user: string
    token?: string
}

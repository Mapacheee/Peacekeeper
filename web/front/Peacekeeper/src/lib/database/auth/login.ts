export type LoginCredentials = {
    email: string
    password: string
}

export type RegisterCredentials = {
    email: string
    password: string
    names: string
    lastnames: string
}

export type LoginResponse = {
    user: string
    token: string
}

export type RegisterResponse = {
    user: string
    token: string
}

export type ErrorResponse = {
    error: string
}

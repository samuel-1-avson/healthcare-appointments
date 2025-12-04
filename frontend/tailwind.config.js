/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            colors: {
                primary: {
                    DEFAULT: '#0ea5e9', // Sky 500
                    hover: '#0284c7', // Sky 600
                    light: '#e0f2fe', // Sky 100
                },
                secondary: {
                    DEFAULT: '#64748b', // Slate 500
                    hover: '#475569', // Slate 600
                },
                accent: {
                    DEFAULT: '#14b8a6', // Teal 500
                    hover: '#0d9488', // Teal 600
                },
                surface: {
                    DEFAULT: '#1e293b', // Slate 800
                    hover: '#334155', // Slate 700
                    active: '#475569', // Slate 600
                },
                background: '#0f172a', // Slate 900
                success: '#22c55e', // Green 500
                warning: '#eab308', // Yellow 500
                error: '#ef4444', // Red 500
            },
            boxShadow: {
                'glass': '0 4px 30px rgba(0, 0, 0, 0.1)',
                'glow': '0 0 15px rgba(14, 165, 233, 0.3)',
            },
            animation: {
                'fade-in': 'fadeIn 0.5s ease-out',
                'slide-in': 'slideIn 0.3s ease-out',
            },
            keyframes: {
                fadeIn: {
                    '0%': {
                        opacity: '0'
                    },
                    '100%': {
                        opacity: '1'
                    },
                },
                slideIn: {
                    '0%': {
                        transform: 'translateX(-20px)',
                        opacity: '0'
                    },
                    '100%': {
                        transform: 'translateX(0)',
                        opacity: '1'
                    },
                },
            },
        },
    },
    plugins: [],
}

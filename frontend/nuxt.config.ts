export default defineNuxtConfig({
  ssr: true,
  devtools: { enabled: true },
  css: ['~/assets/styles/main.css'],
  modules: [],
  app: {
    head: {
      title: 'SFM Filtered Model Viewer',
      meta: [
        {
          name: 'viewport',
          content: 'width=device-width, initial-scale=1'
        }
      ]
    }
  },
  nitro: {
    routeRules: {
      '/api/**': {
        cors: true
      }
    }
  },
  typescript: {
    strict: true,
    typeCheck: false
  },
  compatibilityDate: '2026-03-19'
})

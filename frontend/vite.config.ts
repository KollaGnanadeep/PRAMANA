import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: {
      '/search': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/cases': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});

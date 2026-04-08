import { defineConfig } from 'vite';

export default defineConfig({
  base: './', // For GitHub Pages relative paths
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  }
});

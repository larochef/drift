import { defineConfig } from 'vite'
import { compression } from 'vite-plugin-compression2';

const scalajsMode = process.env.SCALAJS_MODE || "fast";

export default defineConfig({
  root: 'frontend',
  publicDir: process.env.VITE_PUBLIC_DIR || 'public',
  build: {
    outDir: process.env.VITE_OUT_DIR || 'dist',
    emptyOutDir: true,
  },
  plugins: [
    compression({}),
  ],
  server: {
    port: 5173,
    // `ws: true` is not optional: without it Vite registers no upgrade
    // handler for this proxy, the status socket never connects behind the dev
    // server, and every live update in the app is silently dead — a runtime
    // install sits at the state its POST answered with until the page is
    // reloaded (François, 2026-09-20).
    proxy: { '/api': { target: 'http://localhost:4321', ws: true } },
  },
  resolve: {
    alias: [
      {
        find: /^scalajs:(.*)$/,
        replacement: `/../out/frontend/${scalajsMode}LinkJS.dest/$1`
      }
    ]
  }
})

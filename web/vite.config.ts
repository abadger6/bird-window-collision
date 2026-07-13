import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages base path. Repo is served at
// https://<user>.github.io/bird-window-collision/ so every asset URL needs
// the /bird-window-collision/ prefix. `npm run dev` locally serves at /.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/bird-window-collision/' : '/',
  plugins: [react()],
  build: {
    // Build straight into the repo's docs/ folder — GitHub Pages is
    // configured to serve main:/docs, so we retire the separate gh-pages
    // branch and keep source + deployed site on one branch.
    // emptyOutDir: false so we don't wipe the tracked .md files
    // (methodology.md, README.md, sdgsat_*.md, phase2_birdflow.md) that
    // also live under docs/.
    outDir: '../docs',
    emptyOutDir: false,
    // kepler.gl is a big bundle — bump the warning threshold so `npm run
    // build` doesn't scream about chunks over 500 kB. This is expected;
    // there's no reasonable way to code-split kepler further.
    chunkSizeWarningLimit: 4000,
  },
  server: {
    port: 5173,
  },
  // kepler.gl expects a `global` (Node-style) reference in the browser
  // because some of its dependencies (older node-buffer, etc.) reach for it.
  define: {
    global: 'globalThis',
  },
}));

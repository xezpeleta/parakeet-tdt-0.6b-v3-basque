// Service worker to add COOP/COEP headers required for SharedArrayBuffer (WASM threads)
const CACHE_NAME = 'parakeet-basque-asr-v2';

self.addEventListener('install', (e) => {
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (e) => {
  if (e.request.cache === 'only-if-cached' && e.request.mode !== 'same-origin') {
    return;
  }

  // Only inject COOP/COEP into same-origin responses.
  // Cross-origin resources (CDN scripts, WASM, model files) must pass through
  // untouched – their responses are opaque and rebuilding the Response object
  // would strip Content-Type, causing the browser to reject the scripts.
  const isSameOrigin = new URL(e.request.url).origin === self.location.origin;
  if (!isSameOrigin) {
    return; // pass through unchanged
  }

  e.respondWith(
    fetch(e.request)
      .then((response) => {
        if (response.status === 0) return response;

        const newHeaders = new Headers(response.headers);
        newHeaders.set('Cross-Origin-Embedder-Policy', 'credentialless');
        newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });
      })
      .catch(() => fetch(e.request))
  );
});

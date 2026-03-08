// Service worker to add COOP/COEP headers required for SharedArrayBuffer (WASM threads)
const CACHE_NAME = 'parakeet-basque-asr-v1';

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

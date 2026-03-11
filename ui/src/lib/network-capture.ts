interface NetworkError {
  timestamp: string;
  url: string;
  method: string;
  status: number | null;
  statusText: string;
  error: string;
}

const MAX_ERRORS = 100;
const errors: NetworkError[] = [];

let installed = false;

export function installNetworkCapture(): void {
  if (installed) return;
  installed = true;

  const originalFetch = window.fetch;

  window.fetch = async (...args: Parameters<typeof fetch>) => {
    const [input, init] = args;
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const method = init?.method || 'GET';

    try {
      const response = await originalFetch(...args);
      if (!response.ok) {
        errors.push({
          timestamp: new Date().toISOString(),
          url,
          method,
          status: response.status,
          statusText: response.statusText,
          error: `HTTP ${response.status}`,
        });
        if (errors.length > MAX_ERRORS) errors.shift();
      }
      return response;
    } catch (err) {
      errors.push({
        timestamp: new Date().toISOString(),
        url,
        method,
        status: null,
        statusText: '',
        error: err instanceof Error ? err.message : String(err),
      });
      if (errors.length > MAX_ERRORS) errors.shift();
      throw err;
    }
  };
}

export function getNetworkErrors(): NetworkError[] {
  return [...errors];
}

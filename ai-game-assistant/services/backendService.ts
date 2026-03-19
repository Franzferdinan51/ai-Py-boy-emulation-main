import type { GameAction } from '../types';

/**
 * A centralized handler for fetch errors to provide more helpful diagnostics.
 * @param error The caught error.
 * @param context A string describing the action that failed (e.g., 'fetch screen').
 * @returns An Error object with a more informative message.
 */
function handleFetchError(error: unknown, context: string): Error {
    console.error(`Failed to ${context}:`, error);
    if (error instanceof TypeError && error.message === 'Failed to fetch') {
        return new Error(
            'Network connection failed. The app could not reach your Python server.\n\n' +
            'Please check the following common issues:\n\n' +
            '1. **Is the Server Running?**\n' +
            '   - Your server logs indicate it started correctly. Ensure the terminal running the Python server is still open and hasn\'t crashed.\n\n' +
            '2. **Is the Server URL Correct?**\n' +
            '   - Since you are running locally, the URL in Settings should be `http://localhost:5000` or `http://127.0.0.1:5000`.\n' +
            '   - **Do not** use your Tailscale or `192.168.x.x` IP if the browser is on the same PC as the server.\n\n' +
            '3. **Check for Browser "Mixed Content" Errors**\n' +
            '   - This is a very common issue. If this web app is served from an `https://` address, your browser will block requests to an `http://` server by default.\n' +
            '   - **Solution:** Press `F12` to open your browser\'s developer console. Look for an error like "Mixed Content: The page at \'https://...\' was loaded over HTTPS, but requested an insecure resource \'http://...\'. This request has been blocked."\n\n' +
            '4. **Is a Firewall or Antivirus Blocking it?**\n' +
            '   - Your firewall or antivirus software could be blocking the connection, even on `localhost`. Try temporarily disabling them to see if that is the cause.\n\n' +
            '5. **Is CORS Configured?**\n' +
            '   - Ensure your Python server has the `flask-cors` library correctly configured with `CORS(app)`.'
        );
    }
    if (error instanceof Error) {
        return error;
    }
    return new Error(`An unknown error occurred while trying to ${context}.`);
}

/**
 * Checks if the backend server is reachable and responsive.
 * @param baseUrl The base URL of the Python server.
 * @returns A promise that resolves to a success or failure object with a message.
 */
export async function checkBackendStatus(baseUrl: string): Promise<{ success: boolean; message: string; }> {
    try {
        const response = await fetch(`${baseUrl}/api/screen`);
        // A 400 "ROM not loaded" is a success for connection testing purposes.
        if (!response.ok && response.status !== 400) {
            if (response.status === 404) {
                 throw new Error("The backend server does not have a '/api/screen' endpoint. Please ensure the server is running and the URL is correct.");
            }
            throw new Error(`Server responded with status ${response.status}: ${response.statusText}`);
        }
        return { success: true, message: 'Successfully connected to the server.' };
    } catch (error) {
        const detailedError = handleFetchError(error, 'connect to server');
        return { success: false, message: detailedError.message };
    }
}


export async function getScreen(baseUrl: string): Promise<Blob> {
  try {
    const response = await fetch(`${baseUrl}/api/screen`);
    if (!response.ok) {
      throw new Error(`Backend error: ${response.statusText}`);
    }
    return response.blob();
  } catch (error) {
    throw handleFetchError(error, 'fetch screen');
  }
}

export async function sendAction(baseUrl: string, action: GameAction): Promise<void> {
  try {
    // Use /api/game/button endpoint with 'button' parameter
    // GameAction is a string type ('UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'A' | 'B' | 'START' | 'SELECT')
    const response = await fetch(`${baseUrl}/api/game/button`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ button: action }),
    });
    if (!response.ok) {
      throw new Error(`Backend error: ${response.statusText}`);
    }
  } catch (error) {
    console.error(handleFetchError(error, 'send action'));
  }
}

export async function loadRom(baseUrl: string, romFile: File): Promise<void> {
    const formData = new FormData();
    formData.append('rom', romFile);

    try {
        // Use the consistent upload-rom endpoint
        const response = await fetch(`${baseUrl}/api/upload-rom`, {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(errorData.message || 'Failed to load ROM on backend.');
        }
    } catch (error) {
        throw handleFetchError(error, 'upload ROM');
    }
}

export async function saveState(baseUrl: string): Promise<void> {
    try {
        const response = await fetch(`${baseUrl}/api/save_state`, { method: 'POST' });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(errorData.message || 'Failed to save state on backend.');
        }
    } catch (error) {
        throw handleFetchError(error, 'save state');
    }
}

export async function loadState(baseUrl: string): Promise<void> {
    try {
        const response = await fetch(`${baseUrl}/api/load_state`, { method: 'POST' });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(errorData.message || 'Failed to load state on backend.');
        }
    } catch (error) {
        throw handleFetchError(error, 'load state');
    }
}
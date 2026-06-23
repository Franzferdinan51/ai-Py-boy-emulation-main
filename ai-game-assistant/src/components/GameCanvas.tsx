import React, { useCallback, useEffect, useRef, useState } from 'react';

export type StreamingStatus = 'disconnected' | 'connecting' | 'connected' | 'error' | 'failed';

interface GameCanvasProps {
  backendUrl: string;
  onStatusChange?: (status: StreamingStatus) => void;
  className?: string;
  showStats?: boolean;
}

interface StreamStats {
  fps: number;
  frameCount: number;
  bytesReceived: number;
  lastUpdate: number;
}

interface StreamFramePayload {
  image?: string;
  timestamp?: number | string;
  frame?: number;
  fps?: number;
}

const RECONNECT_DELAY_MS = 3000;

const GameCanvas: React.FC<GameCanvasProps> = ({
  backendUrl,
  onStatusChange,
  className = '',
  showStats = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(false);
  const statusRef = useRef<StreamingStatus>('disconnected');
  const onStatusChangeRef = useRef(onStatusChange);
  const connectRef = useRef<() => void>(() => {});

  const [status, setStatus] = useState<StreamingStatus>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<StreamStats>({
    fps: 0,
    frameCount: 0,
    bytesReceived: 0,
    lastUpdate: Date.now(),
  });

  useEffect(() => {
    onStatusChangeRef.current = onStatusChange;
  }, [onStatusChange]);

  const setStreamingStatus = useCallback((nextStatus: StreamingStatus) => {
    statusRef.current = nextStatus;
    setStatus(nextStatus);
    onStatusChangeRef.current?.(nextStatus);
  }, []);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const closeEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const scheduleReconnect = useCallback((message: string) => {
    if (!mountedRef.current || reconnectTimerRef.current !== null) {
      return;
    }

    setError(message);
    setStreamingStatus('error');

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      if (!mountedRef.current) {
        return;
      }
      connectRef.current();
    }, RECONNECT_DELAY_MS);
  }, [setStreamingStatus]);

  const connectToStream = useCallback(() => {
    clearReconnectTimer();
    closeEventSource();

    setError(null);
    setStreamingStatus('connecting');

    const eventSource = new EventSource(`${backendUrl}/api/stream`);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      if (eventSourceRef.current !== eventSource || !mountedRef.current) {
        return;
      }

      setError(null);
      setStreamingStatus('connected');
      console.log('[GameCanvas] Stream connected');
    };

    eventSource.onmessage = (event) => {
      if (eventSourceRef.current !== eventSource || !mountedRef.current) {
        return;
      }

      try {
        const payload = JSON.parse(event.data) as StreamFramePayload;
        if (typeof payload?.image !== 'string' || !payload.image) {
          return;
        }

        const canvas = canvasRef.current;
        if (!canvas) {
          return;
        }

        const context = canvas.getContext('2d');
        if (!context) {
          return;
        }

        const image = new Image();
        image.onload = () => {
          if (eventSourceRef.current !== eventSource || !mountedRef.current) {
            return;
          }

          context.clearRect(0, 0, canvas.width, canvas.height);
          context.drawImage(image, 0, 0, canvas.width, canvas.height);
          setError(null);
          setStats((previous) => ({
            fps: typeof payload.fps === 'number' ? payload.fps : previous.fps,
            frameCount: typeof payload.frame === 'number' ? payload.frame : previous.frameCount + 1,
            bytesReceived: previous.bytesReceived + event.data.length,
            lastUpdate: Date.now(),
          }));

          if (statusRef.current !== 'connected') {
            setStreamingStatus('connected');
          }
        };

        image.onerror = () => {
          if (eventSourceRef.current !== eventSource || !mountedRef.current) {
            return;
          }

          scheduleReconnect('Received an unreadable frame. Retrying...');
        };

        image.src = `data:image/jpeg;base64,${payload.image}`;
      } catch (parseError) {
        console.error('[GameCanvas] Error parsing message:', parseError);
        scheduleReconnect('Received malformed stream data. Retrying...');
      }
    };

    eventSource.onerror = () => {
      if (eventSourceRef.current !== eventSource || !mountedRef.current) {
        return;
      }

      console.error('[GameCanvas] Stream error');
      closeEventSource();
      scheduleReconnect('Connection lost. Retrying...');
    };
  }, [backendUrl, clearReconnectTimer, closeEventSource, scheduleReconnect, setStreamingStatus]);

  useEffect(() => {
    connectRef.current = connectToStream;
  }, [connectToStream]);

  useEffect(() => {
    mountedRef.current = true;
    connectToStream();

    return () => {
      mountedRef.current = false;
      clearReconnectTimer();
      closeEventSource();
    };
  }, [backendUrl, clearReconnectTimer, closeEventSource, connectToStream]);

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return '#00ff88';
      case 'connecting':
        return '#fbbf24';
      case 'error':
      case 'failed':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return showStats ? `LIVE ${stats.fps} FPS` : 'LIVE';
      case 'connecting':
        return 'Connecting...';
      case 'error':
        return 'Connection Error';
      case 'failed':
        return 'Stream Failed';
      default:
        return 'Disconnected';
    }
  };

  return (
    <div className={`relative bg-black rounded-lg overflow-hidden ${className}`}>
      <canvas
        ref={canvasRef}
        width={160}
        height={144}
        className="w-full h-auto pixelated"
        style={{ imageRendering: 'pixelated', aspectRatio: '160/144' }}
      />

      {status !== 'disconnected' && (
        <div className="absolute top-2 right-2 bg-black/80 px-2 py-1 rounded" role="status" aria-live="polite">
          <div className="flex items-center space-x-2">
            <span
              className={`w-2 h-2 rounded-full ${status === 'connected' ? 'animate-pulse' : ''}`}
              style={{ backgroundColor: getStatusColor() }}
            />
            <span className="font-mono text-xs font-bold" style={{ color: getStatusColor() }}>
              {getStatusText()}
            </span>
          </div>
        </div>
      )}

      {(status === 'connecting' || !stats.frameCount) && (
        <div className="absolute inset-0 bg-black/60 flex items-center justify-center" aria-live="polite">
          <div className="flex flex-col items-center space-y-2">
            <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-cyan-400 text-sm">Waiting for game...</span>
          </div>
        </div>
      )}

      {error && status === 'error' && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center p-4" role="alert">
          <div className="text-center">
            <span className="text-red-400 text-lg">⚠️</span>
            <p className="text-red-400 text-sm mt-2">{error}</p>
          </div>
        </div>
      )}

      {showStats && status === 'connected' && (
        <div className="absolute bottom-2 left-2 bg-black/80 px-2 py-1 rounded">
          <div className="font-mono text-xs text-neutral-400 space-x-3">
            <span>Frame: {stats.frameCount}</span>
            <span>{(stats.bytesReceived / 1024).toFixed(1)} KB</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default GameCanvas;

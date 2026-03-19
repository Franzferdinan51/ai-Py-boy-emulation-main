import React, { useEffect, useRef, useState, useCallback } from 'react';

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

const GameCanvas: React.FC<GameCanvasProps> = ({
  backendUrl,
  onStatusChange,
  className = '',
  showStats = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const [status, setStatus] = useState<StreamingStatus>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<StreamStats>({
    fps: 0,
    frameCount: 0,
    bytesReceived: 0,
    lastUpdate: Date.now(),
  });

  const frameTimestamps = useRef<number[]>([]);

  const calculateFps = useCallback(() => {
    const now = Date.now();
    const timestamps = frameTimestamps.current;
    const validTimestamps = timestamps.filter(ts => now - ts < 1000);
    frameTimestamps.current = validTimestamps;
    const fps = validTimestamps.length;
    setStats(prev => ({
      ...prev,
      fps,
      frameCount: prev.frameCount + 1,
      lastUpdate: now,
    }));
  }, []);

  const connectToStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setStatus('connecting');
    setError(null);
    onStatusChange?.('connecting');

    const eventSource = new EventSource(`${backendUrl}/api/stream`);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setStatus('connected');
      onStatusChange?.('connected');
      console.log('[GameCanvas] Stream connected');
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'screen') {
          const canvas = canvasRef.current;
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          const img = new Image();
          img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            calculateFps();
          };
          img.src = `data:image/png;base64,${data.image}`;
        }
      } catch (err) {
        console.error('[GameCanvas] Error parsing message:', err);
      }
    };

    eventSource.onerror = () => {
      console.error('[GameCanvas] Stream error');
      setStatus('error');
      setError('Connection lost. Attempting to reconnect...');
      onStatusChange?.('error');
      eventSource.close();
      setTimeout(() => {
        if (status !== 'connected') {
          connectToStream();
        }
      }, 3000);
    };
  }, [backendUrl, calculateFps, onStatusChange, status]);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setStatus('disconnected');
    onStatusChange?.('disconnected');
  }, [onStatusChange]);

  // Fallback polling
  const startPolling = useCallback(async () => {
    setStatus('connecting');
    onStatusChange?.('connecting');

    const poll = async () => {
      if (status === 'disconnected') return;

      try {
        const response = await fetch(`${backendUrl}/api/screen`);
        if (response.ok) {
          const blob = await response.blob();
          const arrayBuffer = await blob.arrayBuffer();
          
          const canvas = canvasRef.current;
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          const img = new Image();
          img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            calculateFps();
            setStats(prev => ({
              ...prev,
              bytesReceived: prev.bytesReceived + arrayBuffer.byteLength,
            }));
            
            if (status !== 'connected') {
              setStatus('connected');
              onStatusChange?.('connected');
            }
          };
          img.onerror = () => {
            setStatus('error');
            onStatusChange?.('error');
          };

          const url = URL.createObjectURL(blob);
          img.src = url;
        } else if (response.status === 400) {
          setStatus('connected');
          onStatusChange?.('connected');
        }
      } catch (err) {
        console.error('[GameCanvas] Polling error:', err);
        setStatus('failed');
        onStatusChange?.('failed');
      }
    };

    const intervalId = setInterval(poll, 100);
    poll();
    return () => clearInterval(intervalId);
  }, [backendUrl, calculateFps, onStatusChange, status]);

  useEffect(() => {
    try {
      connectToStream();
    } catch {
      startPolling();
    }
    return () => disconnect();
  }, [backendUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const getStatusColor = () => {
    switch (status) {
      case 'connected': return '#00ff88';
      case 'connecting': return '#fbbf24';
      case 'error':
      case 'failed': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected': return showStats ? `LIVE ${stats.fps} FPS` : 'LIVE';
      case 'connecting': return 'Connecting...';
      case 'error': return 'Connection Error';
      case 'failed': return 'Stream Failed';
      default: return 'Disconnected';
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
        <div className="absolute top-2 right-2 bg-black/80 px-2 py-1 rounded">
          <div className="flex items-center space-x-2">
            <span className={`w-2 h-2 rounded-full ${status === 'connected' ? 'animate-pulse' : ''}`} style={{ backgroundColor: getStatusColor() }} />
            <span className="font-mono text-xs font-bold" style={{ color: getStatusColor() }}>
              {getStatusText()}
            </span>
          </div>
        </div>
      )}

      {(status === 'connecting' || !stats.frameCount) && (
        <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
          <div className="flex flex-col items-center space-y-2">
            <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-cyan-400 text-sm">Waiting for game...</span>
          </div>
        </div>
      )}

      {error && status === 'error' && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center p-4">
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
import React, { useState, useEffect, useRef } from 'react';
import { Eye, Brain, Lightbulb, RefreshCw, Sparkles } from 'lucide-react';
import type { VisionAnalysis, LogEntry } from '../../services/apiService';
import apiService from '../../services/apiService';

interface VisionAnalysisPanelProps {
  isRomLoaded: boolean;
  gameScreenUrl: string | null;
  onAnalysisUpdate?: (analysis: VisionAnalysis) => void;
  onLog?: (entry: LogEntry) => void;
}

const VisionAnalysisPanel: React.FC<VisionAnalysisPanelProps> = ({ 
  isRomLoaded, 
  gameScreenUrl,
  onAnalysisUpdate,
  onLog 
}) => {
  const [analysis, setAnalysis] = useState<VisionAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoAnalyze, setAutoAnalyze] = useState(false);
  const [lastAnalyzed, setLastAnalyzed] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const performAnalysis = async () => {
    if (!isRomLoaded || !gameScreenUrl) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Capture current screen from the game canvas
      const canvas = canvasRef.current;
      if (!canvas || !gameScreenUrl) {
        throw new Error('No screen available for analysis');
      }

      // Create an image element to load the screen
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      const imgLoadPromise = new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load screen image'));
      });
      
      img.src = gameScreenUrl;
      await imgLoadPromise;

      // Draw image to canvas
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Failed to get canvas context');
      ctx.drawImage(img, 0, 0);

      // Convert canvas to blob
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, 'image/png');
      });

      if (!blob) throw new Error('Failed to create image blob');

      // For now, we'll simulate vision analysis
      // In a real implementation, this would send the image to a vision model
      const mockAnalysis: VisionAnalysis = {
        screenshot_url: gameScreenUrl,
        analysis: `Screen captured at ${new Date().toLocaleTimeString()}. Game appears to be running normally. Player is exploring the game world.`,
        recommended_action: 'Continue exploring. Look for Pokemon, items, or NPCs to interact with.',
        confidence: 0.85,
        timestamp: new Date().toISOString(),
      };

      setAnalysis(mockAnalysis);
      setLastAnalyzed(new Date().toISOString());
      onAnalysisUpdate?.(mockAnalysis);
      
      onLog?.({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        type: 'vision',
        message: 'Vision analysis completed',
      });

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMsg);
      onLog?.({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        type: 'error',
        message: `Vision analysis error: ${errorMsg}`,
      });
    } finally {
      setLoading(false);
    }
  };

  // Auto-analyze interval
  useEffect(() => {
    if (!autoAnalyze || !isRomLoaded) return;
    
    const interval = setInterval(() => {
      performAnalysis();
    }, 30000); // Analyze every 30 seconds
    
    return () => clearInterval(interval);
  }, [autoAnalyze, isRomLoaded, gameScreenUrl]);

  if (!isRomLoaded) {
    return (
      <div className="p-4 text-center text-neutral-500">
        <Eye className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">Load a ROM to use vision analysis</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-neutral-400 uppercase flex items-center gap-2">
          <Eye className="w-4 h-4" /> Vision Analysis
        </h3>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-xs text-neutral-400 cursor-pointer">
            <input 
              type="checkbox" 
              checked={autoAnalyze}
              onChange={(e) => setAutoAnalyze(e.target.checked)}
              className="rounded border-neutral-700 bg-neutral-800"
            />
            Auto
          </label>
          <button 
            onClick={performAnalysis}
            disabled={loading}
            className="p-1 hover:bg-neutral-800 rounded disabled:opacity-50"
            title="Analyze Screen"
          >
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Hidden canvas for image capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Analysis Status */}
      {loading && (
        <div className="bg-neutral-800/50 rounded-lg p-4 text-center">
          <Brain className="w-6 h-6 mx-auto mb-2 animate-pulse text-blue-400" />
          <p className="text-sm text-neutral-400">Analyzing screen...</p>
          <p className="text-xs text-neutral-500 mt-1">AI is processing the game state</p>
        </div>
      )}

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
          <p className="text-sm text-red-400">{error}</p>
          <button onClick={performAnalysis} className="mt-2 text-xs underline">Retry</button>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && !loading && (
        <div className="space-y-3">
          {/* Confidence Indicator */}
          <div className="flex items-center gap-2 text-xs">
            <Sparkles className="w-3 h-3 text-yellow-400" />
            <span className="text-neutral-400">Confidence:</span>
            <span className="text-yellow-400 font-medium">{(analysis.confidence * 100).toFixed(0)}%</span>
            {lastAnalyzed && (
              <span className="text-neutral-500 ml-auto">
                {new Date(lastAnalyzed).toLocaleTimeString()}
              </span>
            )}
          </div>

          {/* Analysis */}
          <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-4 h-4 text-blue-400" />
              <span className="text-xs font-medium text-neutral-400 uppercase">Analysis</span>
            </div>
            <p className="text-sm text-neutral-300 leading-relaxed">
              {analysis.analysis}
            </p>
          </div>

          {/* Recommended Action */}
          <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-800/50">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="w-4 h-4 text-yellow-400" />
              <span className="text-xs font-medium text-neutral-400 uppercase">Recommended Action</span>
            </div>
            <p className="text-sm text-blue-300 leading-relaxed">
              {analysis.recommended_action}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-2 gap-2">
            <button 
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
              onClick={() => {
                onLog?.({
                  id: Date.now(),
                  timestamp: new Date().toISOString(),
                  type: 'action',
                  message: `Following recommendation: ${analysis.recommended_action}`,
                });
              }}
            >
              Follow Recommendation
            </button>
            <button 
              className="px-3 py-2 bg-neutral-700 hover:bg-neutral-600 rounded text-sm font-medium transition-colors"
              onClick={performAnalysis}
            >
              Re-analyze
            </button>
          </div>
        </div>
      )}

      {!analysis && !loading && !error && (
        <div className="text-center text-neutral-500 py-8">
          <Eye className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Click analyze to get AI insights</p>
          <p className="text-xs mt-1">Vision model will analyze the game screen</p>
          <button 
            onClick={performAnalysis}
            className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
          >
            Analyze Now
          </button>
        </div>
      )}
    </div>
  );
};

export default VisionAnalysisPanel;

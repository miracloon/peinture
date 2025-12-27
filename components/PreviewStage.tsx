
import React, { useRef } from 'react';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { ImageComparison } from './ImageComparison';
import { Paintbrush, AlertCircle, Sparkles, X, Film, Image as ImageIcon } from 'lucide-react';
import { GeneratedImage } from '../types';

interface PreviewStageProps {
    currentImage: GeneratedImage | null;
    isWorking: boolean;
    isTranslating: boolean;
    elapsedTime: number;
    error: string | null;
    onCloseError: () => void;
    isComparing: boolean;
    tempUpscaledImage: string | null;
    showInfo: boolean;
    setShowInfo: (val: boolean) => void;
    imageDimensions: { width: number, height: number } | null;
    setImageDimensions: (val: { width: number, height: number } | null) => void;
    t: any;
    children?: React.ReactNode;
    // New Props for Live
    isLiveMode?: boolean;
    onToggleLiveMode?: () => void;
    isGeneratingVideoPrompt?: boolean;
}

export const PreviewStage: React.FC<PreviewStageProps> = ({
    currentImage,
    isWorking,
    isTranslating,
    elapsedTime,
    error,
    onCloseError,
    isComparing,
    tempUpscaledImage,
    showInfo,
    setShowInfo,
    imageDimensions,
    setImageDimensions,
    t,
    children,
    isLiveMode,
    onToggleLiveMode,
    isGeneratingVideoPrompt
}) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    const isLiveGenerating = currentImage?.videoStatus === 'generating';

    return (
        <section className="relative w-full flex flex-col h-[360px] md:h-[480px] items-center justify-center bg-black/20 rounded-xl backdrop-blur-xl border border-white/10 shadow-2xl shadow-black/20 overflow-hidden relative group">

            {isWorking ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-black/40 backdrop-blur-sm animate-in fade-in duration-500">
                    <div className="relative">
                        <div className="h-24 w-24 rounded-full border-4 border-white/10 border-t-purple-500 animate-spin"></div>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <Paintbrush className="text-purple-400 animate-pulse w-8 h-8" />
                        </div>
                    </div>
                    <p className="mt-8 text-white/80 font-medium animate-pulse text-lg">
                        {isTranslating ? t.translating : t.dreaming}
                    </p>
                    {!isTranslating && (
                        <p className="mt-2 font-mono text-purple-300 text-lg">{elapsedTime.toFixed(1)}s</p>
                    )}
                </div>
            ) : null}

            {error ? (
                <div className="text-center text-red-400 p-8 max-w-md animate-in zoom-in-95 duration-300 relative group/error">
                    <button 
                        onClick={onCloseError}
                        className="absolute -top-2 -right-2 p-2 text-white/40 hover:text-white rounded-full hover:bg-white/10 transition-colors"
                        title={t.close}
                    >
                        <X className="w-5 h-5" />
                    </button>
                    <AlertCircle className="w-16 h-16 mx-auto mb-4 text-red-500/50" />
                    <h3 className="text-xl font-bold text-white mb-2">{t.generationFailed}</h3>
                    <p className="text-white/60">{error}</p>
                </div>
            ) : currentImage ? (
                <div className="w-full h-full flex items-center justify-center bg-black/40 animate-in zoom-in-95 duration-500 relative">

                    {/* Image View, Comparison View, or Video View */}
                    {isComparing && tempUpscaledImage ? (
                        <div className="w-full h-full">
                            <ImageComparison
                                beforeImage={currentImage.url}
                                afterImage={tempUpscaledImage}
                                alt={currentImage.prompt}
                                labelBefore={t.compare_original}
                                labelAfter={t.compare_upscaled}
                            />
                        </div>
                    ) : isLiveMode && currentImage.videoUrl ? (
                        <div className="w-full h-full flex items-center justify-center">
                            <video
                                ref={videoRef}
                                src={currentImage.videoUrl}
                                className={`max-w-full max-h-full object-contain shadow-2xl transition-all duration-300 ${currentImage.isBlurred ? 'blur-lg scale-105' : ''}`}
                                autoPlay
                                loop
                                playsInline
                            />
                        </div>
                    ) : (
                        <TransformWrapper
                            initialScale={1}
                            minScale={1}
                            maxScale={8}
                            centerOnInit={true}
                            key={currentImage.id} // Forces component reset on new image
                            wheel={{ step: 0.5 }}
                        >
                            <TransformComponent
                                wrapperStyle={{ width: "100%", height: "100%" }}
                                contentStyle={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}
                            >
                                <img
                                    src={currentImage.url}
                                    alt={currentImage.prompt}
                                    className={`max-w-full max-h-full object-contain shadow-2xl cursor-grab active:cursor-grabbing transition-all duration-300 ${currentImage.isBlurred ? 'blur-lg scale-105' : ''}`}
                                    onContextMenu={(e) => e.preventDefault()}
                                    onLoad={(e) => {
                                        setImageDimensions({
                                            width: e.currentTarget.naturalWidth,
                                            height: e.currentTarget.naturalHeight
                                        });
                                    }}
                                />
                            </TransformComponent>
                        </TransformWrapper>
                    )}

                    {/* Live Generation Overlay for both Prompt and Video generation phases */}
                    {(isGeneratingVideoPrompt || isLiveGenerating) && !isLiveMode && (
                        <div className="absolute top-4 right-4 bg-black/60 backdrop-blur text-white/80 text-xs px-2 py-1 rounded flex items-center gap-1.5 border border-white/10 z-20">
                            <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"></div>
                            {isGeneratingVideoPrompt ? t.liveGeneratingDesc : t.liveGenerating}
                        </div>
                    )}
                    
                    {/* Live/Image Toggle Button (Top Right, persistent if video exists) */}
                    {currentImage.videoStatus === 'success' && currentImage.videoUrl && !isComparing && (
                         <div className="absolute top-4 right-4 z-20">
                             <button
                                onClick={onToggleLiveMode}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur border border-white/20 text-white/90 hover:bg-white/10 transition-all shadow-lg active:scale-95"
                            >
                                {isLiveMode ? (
                                    <>
                                        <ImageIcon className="w-4 h-4 text-purple-400" />
                                        <span className="text-xs font-medium">Image</span>
                                    </>
                                ) : (
                                    <>
                                        <Film className="w-4 h-4 text-red-400" />
                                        <span className="text-xs font-medium">Live</span>
                                    </>
                                )}
                            </button>
                         </div>
                    )}

                    {children}
                </div>
            ) : !isWorking && (
                <div className="text-center text-white/60 p-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    <div className="relative inline-block">
                        <Sparkles className="w-20 h-20 text-white/10" />
                        <Sparkles className="w-20 h-20 text-purple-500/40 absolute top-0 left-0 blur-lg animate-pulse" />
                    </div>
                    <h2 className="mt-6 text-2xl font-bold text-white/90">{t.galleryEmptyTitle}</h2>
                    <p className="mt-2 text-base text-white/40 max-w-xs mx-auto">{t.galleryEmptyDesc}</p>
                </div>
            )}
        </section>
    );
};

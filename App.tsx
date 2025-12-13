
import React, { useState, useEffect, useRef } from 'react';
import { generateImage, optimizePrompt, upscaler } from './services/hfService';
import { generateGiteeImage, optimizePromptGitee } from './services/giteeService';
import { generateMSImage, optimizePromptMS } from './services/msService';
import { translatePrompt } from './services/utils';
import { GeneratedImage, AspectRatioOption, ModelOption, ProviderOption } from './types';
import { HistoryGallery } from './components/HistoryGallery';
import { SettingsModal } from './components/SettingsModal';
import { FAQModal } from './components/FAQModal';
import { Logo } from './components/Icons';
import { Tooltip } from './components/Tooltip';
import { translations, Language } from './translations';
import {
  Sparkles,
  Loader2,
  Settings,
  RotateCcw,
  CircleHelp,
  Github,
} from 'lucide-react';
import { getModelConfig, getGuidanceScaleConfig, FLUX_MODELS, HF_MODEL_OPTIONS, GITEE_MODEL_OPTIONS, MS_MODEL_OPTIONS } from './constants';
import { PromptInput } from './components/PromptInput';
import { ControlPanel } from './components/ControlPanel';
import { PreviewStage } from './components/PreviewStage';
import { ImageToolbar } from './components/ImageToolbar';

export default function App() {
  // Language Initialization
  const [lang, setLang] = useState<Language>(() => {
    const saved = localStorage.getItem('app_language');
    if (saved === 'en' || saved === 'zh') return saved;
    const browserLang = navigator.language.toLowerCase();
    return browserLang.startsWith('zh') ? 'zh' : 'en';
  });
  
  const t = translations[lang];

  // Dynamic Aspect Ratio Options based on language
  const aspectRatioOptions = [
    { value: '1:1', label: t.ar_square },
    { value: '9:16', label: t.ar_photo_9_16 },
    { value: '16:9', label: t.ar_movie },
    { value: '3:4', label: t.ar_portrait_3_4 },
    { value: '4:3', label: t.ar_landscape_4_3 },
    { value: '3:2', label: t.ar_portrait_3_2 },
    { value: '2:3', label: t.ar_landscape_2_3 },
  ];

  const [prompt, setPrompt] = useState<string>('');
  const [provider, setProvider] = useState<ProviderOption>('huggingface');
  const [model, setModel] = useState<ModelOption>('z-image-turbo');
  const [aspectRatio, setAspectRatio] = useState<AspectRatioOption>('1:1');
  const [seed, setSeed] = useState<string>(''); 
  const [steps, setSteps] = useState<number>(9);
  const [guidanceScale, setGuidanceScale] = useState<number>(3.5);
  const [enableHD, setEnableHD] = useState<boolean>(false);
  const [autoTranslate, setAutoTranslate] = useState<boolean>(false);
  
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isTranslating, setIsTranslating] = useState<boolean>(false);
  const [isOptimizing, setIsOptimizing] = useState<boolean>(false);
  const [isUpscaling, setIsUpscaling] = useState<boolean>(false);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);
  const [currentImage, setCurrentImage] = useState<GeneratedImage | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Transition state for upscaling
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [tempUpscaledImage, setTempUpscaledImage] = useState<string | null>(null);
  
  // Initialize history from localStorage with expiration check (delete older than 1 day)
  const [history, setHistory] = useState<GeneratedImage[]>(() => {
    try {
      const saved = localStorage.getItem('ai_image_gen_history');
      if (!saved) return [];
      
      const parsedHistory: GeneratedImage[] = JSON.parse(saved);
      const now = Date.now();
      const oneDayInMs = 24 * 60 * 60 * 1000;
      
      // Filter out images older than 1 day
      return parsedHistory.filter(img => (now - img.timestamp) < oneDayInMs);
    } catch (e) {
      console.error("Failed to load history", e);
      return [];
    }
  });

  const [error, setError] = useState<string | null>(null);
  
  // New state for Info Popover
  const [showInfo, setShowInfo] = useState<boolean>(false);
  const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null);
  const [copiedPrompt, setCopiedPrompt] = useState<boolean>(false);

  // Settings State
  const [showSettings, setShowSettings] = useState<boolean>(false);
  
  // FAQ State
  const [showFAQ, setShowFAQ] = useState<boolean>(false);

  // Language Persistence
  useEffect(() => {
    localStorage.setItem('app_language', lang);
  }, [lang]);

  // Image History Persistence
  useEffect(() => {
    localStorage.setItem('ai_image_gen_history', JSON.stringify(history));
  }, [history]);

  // Update steps and guidance scale when model/provider changes
  useEffect(() => {
      const config = getModelConfig(provider, model);
      setSteps(config.default);

      const gsConfig = getGuidanceScaleConfig(model, provider);
      if (gsConfig) {
          setGuidanceScale(gsConfig.default);
      }
  }, [provider, model]);

  // Handle Auto Translate default state based on model
  useEffect(() => {
    if (FLUX_MODELS.includes(model)) {
        setAutoTranslate(true);
    } else {
        setAutoTranslate(false);
    }
  }, [model]);

  // Initial Selection Effect
  useEffect(() => {
    if (!currentImage && history.length > 0) {
      setCurrentImage(history[0]);
    }
  }, [history.length]); 

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
        if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startTimer = () => {
    setElapsedTime(0);
    const startTime = Date.now();
    timerRef.current = setInterval(() => {
        setElapsedTime((Date.now() - startTime) / 1000);
    }, 100);
    return startTime;
  };

  const stopTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const addToPromptHistory = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    
    // Read current history from session storage
    let currentHistory: string[] = [];
    try {
        const saved = sessionStorage.getItem('prompt_history');
        currentHistory = saved ? JSON.parse(saved) : [];
    } catch (e) {}

    // Update
    const filtered = currentHistory.filter(p => p !== trimmed);
    const newHistory = [trimmed, ...filtered].slice(0, 50);

    // Save
    sessionStorage.setItem('prompt_history', JSON.stringify(newHistory));
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    addToPromptHistory(prompt);

    setIsLoading(true);
    setError(null);
    setShowInfo(false); 
    setImageDimensions(null);
    setIsComparing(false);
    setTempUpscaledImage(null);
    
    let finalPrompt = prompt;

    // Handle Auto Translate
    if (autoTranslate) {
        setIsTranslating(true);
        try {
            finalPrompt = await translatePrompt(prompt);
            setPrompt(finalPrompt); // Update UI with translated text
        } catch (err: any) {
            console.error("Translation failed", err);
        } finally {
            setIsTranslating(false);
        }
    }

    const startTime = startTimer();

    try {
      const seedNumber = seed.trim() === '' ? undefined : parseInt(seed, 10);
      const gsConfig = getGuidanceScaleConfig(model, provider);
      const currentGuidanceScale = gsConfig ? guidanceScale : undefined;

      let result;

      if (provider === 'gitee') {
         result = await generateGiteeImage(model, finalPrompt, aspectRatio, seedNumber, steps, enableHD, currentGuidanceScale);
      } else if (provider === 'modelscope') {
         result = await generateMSImage(model, finalPrompt, aspectRatio, seedNumber, steps, enableHD, currentGuidanceScale);
      } else {
         result = await generateImage(model, finalPrompt, aspectRatio, seedNumber, enableHD, steps, currentGuidanceScale);
      }
      
      const endTime = Date.now();
      const duration = (endTime - startTime) / 1000;
      
      const newImage = { 
          ...result, 
          duration, 
          provider, 
          guidanceScale: currentGuidanceScale 
      };
      
      setCurrentImage(newImage);
      setHistory(prev => [newImage, ...prev]);
    } catch (err: any) {
      const errorMessage = (t as any)[err.message] || err.message || t.generationFailed;
      setError(errorMessage);
    } finally {
      stopTimer();
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPrompt('');
    if (provider === 'gitee') {
        setModel(GITEE_MODEL_OPTIONS[0].value as ModelOption);
    } else if (provider === 'modelscope') {
        setModel(MS_MODEL_OPTIONS[0].value as ModelOption);
    } else {
        setModel(HF_MODEL_OPTIONS[0].value as ModelOption);
    }
    setAspectRatio('1:1');
    setSeed('');
    const config = getModelConfig(provider, model);
    setSteps(config.default);
    setEnableHD(false);
    setCurrentImage(null);
    setIsComparing(false);
    setTempUpscaledImage(null);
    setError(null);
  };

  const handleUpscale = async () => {
    if (!currentImage || isUpscaling) return;
    setIsUpscaling(true);
    setError(null);
    try {
        const { url: newUrl } = await upscaler(currentImage.url);
        setTempUpscaledImage(newUrl);
        setIsComparing(true);
    } catch (err: any) {
        setTempUpscaledImage(null);
        const errorMessage = (t as any)[err.message] || err.message || t.error_upscale_failed;
        setError(errorMessage);
    } finally {
        setIsUpscaling(false);
    }
  };

  const handleApplyUpscale = () => {
    if (!currentImage || !tempUpscaledImage) return;
    const updatedImage = { 
        ...currentImage, 
        url: tempUpscaledImage, 
        isUpscaled: true 
    };
    setCurrentImage(updatedImage);
    setHistory(prev => prev.map(img => 
        img.id === updatedImage.id ? updatedImage : img
    ));
    setIsComparing(false);
    setTempUpscaledImage(null);
  };

  const handleCancelUpscale = () => {
    setIsComparing(false);
    setTempUpscaledImage(null);
  };

  const handleOptimizePrompt = async () => {
    if (!prompt.trim()) return;
    addToPromptHistory(prompt);
    setIsOptimizing(true);
    setError(null);
    try {
        let optimized = '';
        if (provider === 'gitee') {
             optimized = await optimizePromptGitee(prompt, lang);
        } else if (provider === 'modelscope') {
             optimized = await optimizePromptMS(prompt, lang);
        } else {
             optimized = await optimizePrompt(prompt, lang);
        }
        setPrompt(optimized);
    } catch (err: any) {
        console.error("Optimization failed", err);
        const errorMessage = (t as any)[err.message] || err.message || t.error_prompt_optimization_failed;
        setError(errorMessage);
    } finally {
        setIsOptimizing(false);
    }
  };

  const handleHistorySelect = (image: GeneratedImage) => {
    setCurrentImage(image);
    setShowInfo(false); 
    setImageDimensions(null); 
    setIsComparing(false);
    setTempUpscaledImage(null);
    setError(null);
  };

  const handleDelete = () => {
    if (!currentImage) return;
    const newHistory = history.filter(img => img.id !== currentImage.id);
    setHistory(newHistory);
    if (newHistory.length > 0) {
      setCurrentImage(newHistory[0]);
    } else {
      setCurrentImage(null);
    }
    setShowInfo(false);
    setIsComparing(false);
    setTempUpscaledImage(null);
    setError(null);
  };

  const handleToggleBlur = () => {
    if (!currentImage) return;
    const newStatus = !currentImage.isBlurred;
    const updatedImage = { ...currentImage, isBlurred: newStatus };
    setCurrentImage(updatedImage);
    setHistory(prev => prev.map(img => 
      img.id === currentImage.id ? updatedImage : img
    ));
  };

  const handleCopyPrompt = async () => {
    if (!currentImage?.prompt) return;
    try {
      await navigator.clipboard.writeText(currentImage.prompt);
      setCopiedPrompt(true);
      setTimeout(() => setCopiedPrompt(false), 2000);
    } catch (err) {
      console.error("Failed to copy", err);
    }
  };

  const handleDownload = async (imageUrl: string, fileName: string) => {
      // (Simplified logic for brevity, moving core logic out is better but for now we keep it here as requested structure)
      // Note: Reusing the same download logic as before
    if (isDownloading) return;
    setIsDownloading(true);

    try {
      const isWebPUrl = imageUrl.toLowerCase().split('?')[0].endsWith('.webp');
      const isWebPData = imageUrl.startsWith('data:image/webp');
      const shouldConvert = isWebPUrl || isWebPData;

      let converted = false;
      if (shouldConvert) {
        try {
          await new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.src = imageUrl;
            img.onload = () => {
              try {
                const canvas = document.createElement('canvas');
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                const ctx = canvas.getContext('2d');
                if (!ctx) { reject(new Error('Canvas context not found')); return; }
                ctx.drawImage(img, 0, 0);
                canvas.toBlob((blob) => {
                  if (!blob) { reject(new Error('Canvas serialization failed')); return; }
                  const url = window.URL.createObjectURL(blob);
                  const link = document.createElement('a');
                  link.href = url;
                  let safeFileName = fileName.replace(/\.webp$/i, '');
                  if (!safeFileName.toLowerCase().endsWith('.png')) { safeFileName += '.png'; }
                  link.download = safeFileName;
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                  window.URL.revokeObjectURL(url);
                  resolve(true);
                }, 'image/png');
              } catch (err) { reject(err); }
            };
            img.onerror = (e) => reject(new Error('Image load failed'));
          });
          converted = true;
        } catch (conversionError) {
          console.warn("PNG conversion failed, falling back", conversionError);
        }
      }

      if (!converted) {
        if (imageUrl.startsWith('data:')) {
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else {
            const response = await fetch(imageUrl, { mode: 'cors' });
            if (!response.ok) throw new Error('Network response was not ok');
            const blob = await response.blob();
            const blobUrl = window.URL.createObjectURL(blob);
            let extension = 'png';
            if (blob.type) {
                const typeParts = blob.type.split('/');
                if (typeParts.length > 1) extension = typeParts[1];
            }
            const finalFileName = fileName.includes('.') ? fileName : `${fileName}.${extension}`;
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = finalFileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(blobUrl);
        }
      }
    } catch (e) {
      console.error("All download methods failed:", e);
      window.open(imageUrl, '_blank');
    } finally {
        setIsDownloading(false);
    }
  };

  const isWorking = isLoading;

  return (
    <div className="relative flex h-auto min-h-screen w-full flex-col overflow-x-hidden bg-gradient-brilliant">
      <div className="flex h-full grow flex-col">
        {/* Header */}
        <header className="w-full backdrop-blur-md sticky top-0 z-50 bg-background-dark/30 border-b border-white/5">
          <div className="max-w-7xl mx-auto flex items-center justify-between px-4 py-3 md:px-8 md:py-4">
            <div className="flex items-center gap-2 text-white">
              <Logo className="size-10" />
              <h1 className="text-white text-xl font-bold leading-tight tracking-[-0.015em]">{t.appTitle}</h1>
            </div>
            
            <div className="flex gap-1">
              <Tooltip content={t.sourceCode} position="bottom">
                  <a
                    href="https://github.com/Amery2010/peinture"
                    className="flex items-center justify-center p-2 rounded-lg text-white/70 hover:text-purple-400 hover:bg-white/10 transition-all active:scale-95"
                    target="_blank"
                  >
                    <Github className="w-5 h-5" />
                  </a>
              </Tooltip>

              <Tooltip content={t.help} position="bottom">
                  <button
                    onClick={() => setShowFAQ(true)}
                    className="flex items-center justify-center p-2 rounded-lg text-white/70 hover:text-green-400 hover:bg-white/10 transition-all active:scale-95"
                  >
                    <CircleHelp className="w-5 h-5" />
                  </button>
              </Tooltip>

              <Tooltip content={t.settings} position="bottom">
                  <button
                    onClick={() => setShowSettings(true)}
                    className="flex items-center justify-center p-2 rounded-lg text-white/70 hover:text-purple-400 hover:bg-white/10 transition-all active:scale-95"
                  >
                    <Settings className="w-5 h-5" />
                  </button>
              </Tooltip>
            </div>
          </div>
        </header>

        <main className="w-full max-w-7xl flex-1 flex flex-col-reverse md:items-stretch md:mx-auto md:flex-row gap-4 md:gap-6 px-4 md:px-8 pb-4 md:pb-8 pt-4 md:pt-6">
          
          {/* Left Column: Controls */}
          <aside className="w-full md:max-w-sm flex-shrink-0 flex flex-col gap-4 md:gap-6">
            <div className="flex-grow space-y-4 md:space-y-6">
              <div className="relative z-10 bg-black/20 p-4 md:p-6 rounded-xl backdrop-blur-xl border border-white/10 flex flex-col gap-4 md:gap-6 shadow-2xl shadow-black/20">
                
                {/* Prompt Input Component */}
                <PromptInput 
                    prompt={prompt}
                    setPrompt={setPrompt}
                    isOptimizing={isOptimizing}
                    onOptimize={handleOptimizePrompt}
                    isTranslating={isTranslating}
                    autoTranslate={autoTranslate}
                    setAutoTranslate={setAutoTranslate}
                    t={t}
                    addToPromptHistory={addToPromptHistory}
                />

                {/* Control Panel Component */}
                <ControlPanel 
                    provider={provider}
                    setProvider={setProvider}
                    model={model}
                    setModel={setModel}
                    aspectRatio={aspectRatio}
                    setAspectRatio={setAspectRatio}
                    steps={steps}
                    setSteps={setSteps}
                    guidanceScale={guidanceScale}
                    setGuidanceScale={setGuidanceScale}
                    seed={seed}
                    setSeed={setSeed}
                    enableHD={enableHD}
                    setEnableHD={setEnableHD}
                    t={t}
                    aspectRatioOptions={aspectRatioOptions}
                />
              </div>

              {/* Generate Button & Reset Button */}
              <div className="flex items-center gap-3">
                <button 
                    onClick={handleGenerate}
                    disabled={isWorking || !prompt.trim() || isTranslating}
                    className="group relative flex-1 flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-12 px-4 text-white text-lg font-bold leading-normal tracking-[0.015em] transition-all shadow-lg shadow-purple-900/40 generate-button-gradient hover:shadow-purple-700/50 disabled:opacity-70 disabled:cursor-not-allowed disabled:grayscale"
                >
                    {isLoading || isTranslating ? (
                    <div className="flex items-center gap-2">
                        <Loader2 className="animate-spin w-5 h-5" />
                        <span>{isTranslating ? t.translating : t.dreaming}</span>
                    </div>
                    ) : (
                    <span className="flex items-center gap-2">
                        <Sparkles className="w-5 h-5 group-hover:animate-pulse" />
                        <span className="truncate">{t.generate}</span>
                    </span>
                    )}
                </button>

                {currentImage && (
                    <Tooltip content={t.reset}>
                        <button 
                            onClick={handleReset}
                            className="flex-shrink-0 flex items-center justify-center w-12 h-12 rounded-xl bg-white/5 border border-white/10 text-white/70 hover:text-white hover:bg-white/10 hover:border-white/20 transition-all shadow-lg active:scale-95"
                        >
                            <RotateCcw className="w-5 h-5" />
                        </button>
                    </Tooltip>
                )}
              </div>

            </div>
          </aside>

          {/* Right Column: Preview & Gallery */}
          <div className="flex-1 flex flex-col flex-grow overflow-x-hidden">
            
            {/* Main Preview Area */}
            <PreviewStage 
                currentImage={currentImage}
                isWorking={isWorking}
                isTranslating={isTranslating}
                elapsedTime={elapsedTime}
                error={error}
                isComparing={isComparing}
                tempUpscaledImage={tempUpscaledImage}
                showInfo={showInfo}
                setShowInfo={setShowInfo}
                imageDimensions={imageDimensions}
                setImageDimensions={setImageDimensions}
                t={t}
                copiedPrompt={copiedPrompt}
                handleCopyPrompt={handleCopyPrompt}
            >
                <ImageToolbar 
                    currentImage={currentImage}
                    isComparing={isComparing}
                    showInfo={showInfo}
                    setShowInfo={setShowInfo}
                    isUpscaling={isUpscaling}
                    isDownloading={isDownloading}
                    handleUpscale={handleUpscale}
                    handleToggleBlur={handleToggleBlur}
                    handleDownload={() => currentImage && handleDownload(currentImage.url, `generated-${currentImage.id}`)}
                    handleDelete={handleDelete}
                    handleCancelUpscale={handleCancelUpscale}
                    handleApplyUpscale={handleApplyUpscale}
                    t={t}
                />
            </PreviewStage>

            {/* Gallery Strip */}
            <HistoryGallery 
                images={history} 
                onSelect={handleHistorySelect} 
                selectedId={currentImage?.id}
            />

          </div>
        </main>
        
        {/* Settings Modal */}
        <SettingsModal 
            isOpen={showSettings} 
            onClose={() => setShowSettings(false)} 
            lang={lang}
            setLang={setLang}
            t={t}
            provider={provider}
        />

        {/* FAQ Modal */}
        <FAQModal 
            isOpen={showFAQ}
            onClose={() => setShowFAQ(false)}
            t={t}
        />
      </div>
    </div>
  );
}

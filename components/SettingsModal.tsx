
import React, { useState, useEffect } from 'react';
import { X, Save, KeyRound, Languages, ShieldCheck, ShieldAlert, Database, Eye, EyeOff, MessageSquare, RotateCcw, Settings2, MessageSquareText, Brain, Film, Clock, Layers, Sparkles, HardDrive, Server, Loader2, Check, AlertCircle, PlugZap } from 'lucide-react';
import { Language } from '../translations';
import { getTokenStats } from '../services/hfService';
import { getGiteeTokenStats } from '../services/giteeService';
import { getMsTokenStats } from '../services/msService';
import { ProviderOption, S3Config, WebDAVConfig, StorageType } from '../types';
import { 
    getSystemPromptContent,
    saveSystemPromptContent,
    DEFAULT_SYSTEM_PROMPT_CONTENT,
    getOptimizationModel,
    saveOptimizationModel,
    DEFAULT_OPTIMIZATION_MODELS,
    getVideoSettings,
    saveVideoSettings,
    DEFAULT_VIDEO_SETTINGS,
    VideoSettings
} from '../services/utils';
import { 
    getS3Config, 
    saveS3Config, 
    DEFAULT_S3_CONFIG,
    getWebDAVConfig,
    saveWebDAVConfig,
    DEFAULT_WEBDAV_CONFIG,
    getStorageType,
    saveStorageType,
    testWebDAVConnection,
    testS3Connection
} from '../services/storageService';
import { Select } from './Select';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    lang: Language;
    setLang: (lang: Language) => void;
    t: any;
    provider: ProviderOption;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, lang, setLang, t, provider }) => {
    // Tab State
    const [activeTab, setActiveTab] = useState<'general' | 'prompt' | 'live' | 's3' | 'webdav'>('general');

    // HF Token State
    const [token, setToken] = useState('');
    const [stats, setStats] = useState({ total: 0, active: 0, exhausted: 0 });
    const [showToken, setShowToken] = useState(false);

    // Gitee Token State
    const [giteeToken, setGiteeToken] = useState('');
    const [giteeStats, setGiteeStats] = useState({ total: 0, active: 0, exhausted: 0 });
    const [showGiteeToken, setShowGiteeToken] = useState(false);

    // Model Scope Token State
    const [msToken, setMsToken] = useState('');
    const [msStats, setMsStats] = useState({ total: 0, active: 0, exhausted: 0 });
    const [showMsToken, setShowMsToken] = useState(false);

    // System Prompt State
    const [systemPrompt, setSystemPrompt] = useState('');

    // Optimization Model State
    const [optimModel, setOptimModel] = useState('');

    // Video Settings State
    const [videoSettings, setVideoSettings] = useState<VideoSettings>(DEFAULT_VIDEO_SETTINGS['huggingface']);

    // Storage Config State
    const [storageType, setStorageType] = useState<StorageType>('off');
    const [s3Config, setS3Config] = useState<S3Config>(DEFAULT_S3_CONFIG);
    const [showS3Secret, setShowS3Secret] = useState(false);
    const [webdavConfig, setWebdavConfig] = useState<WebDAVConfig>(DEFAULT_WEBDAV_CONFIG);
    const [showWebdavPass, setShowWebdavPass] = useState(false);
    
    // WebDAV Test State
    const [isTestingWebDAV, setIsTestingWebDAV] = useState(false);
    const [testWebDAVResult, setTestWebDAVResult] = useState<{ success: boolean; message: string } | null>(null);

    // S3 Test State
    const [isTestingS3, setIsTestingS3] = useState(false);
    const [testS3Result, setTestS3Result] = useState<{ success: boolean; message: string } | null>(null);

    useEffect(() => {
        if (isOpen) {
            // Load HF
            const storedToken = localStorage.getItem('huggingFaceToken') || '';
            setToken(storedToken);
            setStats(getTokenStats(storedToken));

            // Load Gitee
            const storedGiteeToken = localStorage.getItem('giteeToken') || '';
            setGiteeToken(storedGiteeToken);
            setGiteeStats(getGiteeTokenStats(storedGiteeToken));

            // Load Model Scope
            const storedMsToken = localStorage.getItem('msToken') || '';
            setMsToken(storedMsToken);
            setMsStats(getMsTokenStats(storedMsToken));

            // Load System Prompt
            setSystemPrompt(getSystemPromptContent());

            // Load Optimization Model for current provider
            setOptimModel(getOptimizationModel(provider));

            // Load Video Settings for current provider
            setVideoSettings(getVideoSettings(provider));

            // Load Storage Config
            setStorageType(getStorageType());
            setS3Config(getS3Config());
            setWebdavConfig(getWebDAVConfig());
            
            // Reset test state
            setTestWebDAVResult(null);
            setTestS3Result(null);
        }
    }, [isOpen, provider]);

    // HF Handlers
    const handleTokenChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newVal = e.target.value;
        setToken(newVal);
        setStats(getTokenStats(newVal));
    };

    const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
        const text = e.clipboardData.getData('text');
        if (text.includes('\n') || text.includes('\r')) {
            e.preventDefault();
            const normalized = text.split(/[\r\n]+/).map(t => t.trim()).filter(Boolean).join(',');
            
            const input = e.currentTarget;
            const start = input.selectionStart ?? token.length;
            const end = input.selectionEnd ?? token.length;
            
            const newValue = token.substring(0, start) + normalized + token.substring(end);
            setToken(newValue);
            setStats(getTokenStats(newValue));
        }
    };

    // Gitee Handlers
    const handleGiteeTokenChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newVal = e.target.value;
        setGiteeToken(newVal);
        setGiteeStats(getGiteeTokenStats(newVal));
    };

    const handleGiteePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
        const text = e.clipboardData.getData('text');
        if (text.includes('\n') || text.includes('\r')) {
            e.preventDefault();
            const normalized = text.split(/[\r\n]+/).map(t => t.trim()).filter(Boolean).join(',');
            
            const input = e.currentTarget;
            const start = input.selectionStart ?? giteeToken.length;
            const end = input.selectionEnd ?? giteeToken.length;
            
            const newValue = giteeToken.substring(0, start) + normalized + giteeToken.substring(end);
            setGiteeToken(newValue);
            setGiteeStats(getGiteeTokenStats(newValue));
        }
    };

    // Model Scope Handlers
    const handleMsTokenChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newVal = e.target.value;
        setMsToken(newVal);
        setMsStats(getMsTokenStats(newVal));
    };

    const handleMsPaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
        const text = e.clipboardData.getData('text');
        if (text.includes('\n') || text.includes('\r')) {
            e.preventDefault();
            const normalized = text.split(/[\r\n]+/).map(t => t.trim()).filter(Boolean).join(',');
            
            const input = e.currentTarget;
            const start = input.selectionStart ?? msToken.length;
            const end = input.selectionEnd ?? msToken.length;
            
            const newValue = msToken.substring(0, start) + normalized + msToken.substring(end);
            setMsToken(newValue);
            setMsStats(getMsTokenStats(newValue));
        }
    };

    // Video Settings Handlers
    const handleRestoreVideoDefaults = () => {
        setVideoSettings(DEFAULT_VIDEO_SETTINGS[provider] || DEFAULT_VIDEO_SETTINGS['huggingface']);
    };

    const handleSave = () => {
        localStorage.setItem('huggingFaceToken', token.trim());
        localStorage.setItem('giteeToken', giteeToken.trim());
        localStorage.setItem('msToken', msToken.trim());
        
        saveSystemPromptContent(systemPrompt);
        saveOptimizationModel(provider, optimModel);
        saveVideoSettings(provider, videoSettings);
        
        saveStorageType(storageType);
        saveS3Config(s3Config);
        saveWebDAVConfig(webdavConfig);
        
        // Dispatch storage event to notify components
        window.dispatchEvent(new Event("storage"));
        
        onClose();
    };

    const handleRestoreDefault = () => {
        setSystemPrompt(DEFAULT_SYSTEM_PROMPT_CONTENT);
        setOptimModel(DEFAULT_OPTIMIZATION_MODELS[provider]);
    };
    
    const handleTestWebDAV = async () => {
        setIsTestingWebDAV(true);
        setTestWebDAVResult(null);
        try {
            const result = await testWebDAVConnection(webdavConfig);
            setTestWebDAVResult(result);
        } catch (e) {
            setTestWebDAVResult({ success: false, message: "Unknown error occurred" });
        } finally {
            setIsTestingWebDAV(false);
        }
    };

    const handleTestS3 = async () => {
        setIsTestingS3(true);
        setTestS3Result(null);
        try {
            const result = await testS3Connection(s3Config);
            setTestS3Result(result);
        } catch (e) {
            setTestS3Result({ success: false, message: "Unknown error occurred" });
        } finally {
            setIsTestingS3(false);
        }
    };

    const getEndpointPlaceholder = () => {
         const region = s3Config.region || 'us-east-1';
         return `https://s3.${region}.amazonaws.com`;
    };

    // Tabs Config
    const tabs = [
        { id: 'general', icon: Settings2, label: t.tab_general },
        { id: 'prompt', icon: MessageSquareText, label: t.tab_prompt },
        { id: 'live', icon: Film, label: t.tab_live },
    ];

    if (storageType === 's3') {
        tabs.push({ id: 's3', icon: HardDrive, label: t.tab_storage });
    } else if (storageType === 'webdav') {
        tabs.push({ id: 'webdav', icon: Server, label: t.tab_webdav });
    }

    const activeTabIndex = tabs.findIndex(tab => tab.id === activeTab);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center px-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
             <div className="w-full max-w-md bg-[#0D0B14]/95 backdrop-blur-xl border border-white/[0.08] rounded-2xl shadow-[0_0_50px_-12px_rgba(124,58,237,0.15)] ring-1 ring-white/[0.05] overflow-hidden animate-in zoom-in-95 duration-300 flex flex-col max-h-[90vh]">
                <div className="flex items-center justify-between px-5 py-2 border-b border-white/[0.06] bg-white/[0.02] flex-shrink-0">
                    <h2 className="text-lg font-bold text-white tracking-wide">{t.settings}</h2>
                    <button onClick={onClose} className="group p-2 rounded-lg text-white/40 hover:text-white hover:bg-white/[0.08] transition-all duration-200">
                        <X className="w-5 h-5 transition-transform duration-500 ease-out group-hover:rotate-180" />
                    </button>
                </div>

                {/* Tab Navigation */}
                <div className="flex items-center px-5 border-b border-white/[0.06] space-x-6 flex-shrink-0 overflow-x-auto scrollbar-hide">
                    {tabs.map((tab) => (
                        <button 
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`group relative py-4 text-sm font-medium transition-colors duration-300 flex items-center gap-2 flex-shrink-0 ${activeTab === tab.id ? 'text-white' : 'text-white/40 hover:text-white/80'}`}
                        >
                            <tab.icon className={`w-4 h-4 transition-colors duration-300 ${activeTab === tab.id ? 'text-purple-400' : 'text-current group-hover:text-purple-400/70'}`} />
                            {tab.label}
                            <span className={`absolute bottom-0 left-0 w-full h-0.5 bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full shadow-[0_-2px_10px_rgba(168,85,247,0.6)] transition-all duration-300 ease-out origin-center ${activeTab === tab.id ? 'opacity-100 scale-x-100' : 'opacity-0 scale-x-0'}`} />
                        </button>
                    ))}
                </div>
                
                {/* Tab Content Container */}
                <div className="flex-1 overflow-hidden relative">
                    <div 
                        className="flex h-full transition-transform duration-500 ease-in-out"
                        style={{ transform: `translateX(-${activeTabIndex * 100}%)` }}
                    >
                        {tabs.map((tab) => (
                            <div key={tab.id} className="w-full h-full flex-shrink-0 overflow-y-auto custom-scrollbar p-5 h-[300px]">
                                {/* Tab 1: General */}
                                {tab.id === 'general' && (
                                    <div className="space-y-5">
                                        {/* Language Selector */}
                                        <div>
                                            <label className="flex items-center gap-2 text-xs font-medium text-white/80 mb-2">
                                                <Languages className="w-3.5 h-3.5 text-purple-400" />
                                                {t.language}
                                            </label>
                                            <div className="grid grid-cols-2 gap-2">
                                                <button
                                                    onClick={() => setLang('en')}
                                                    className={`px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 border ${
                                                        lang === 'en' 
                                                        ? 'bg-purple-600/90 border-purple-500/50 text-white shadow-lg shadow-purple-900/20' 
                                                        : 'bg-white/[0.03] border-white/10 text-white/60 hover:bg-white/[0.06] hover:text-white hover:border-white/20'
                                                    }`}
                                                >
                                                    English
                                                </button>
                                                <button
                                                    onClick={() => setLang('zh')}
                                                    className={`px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 border ${
                                                        lang === 'zh' 
                                                        ? 'bg-purple-600/90 border-purple-500/50 text-white shadow-lg shadow-purple-900/20' 
                                                        : 'bg-white/[0.03] border-white/10 text-white/60 hover:bg-white/[0.06] hover:text-white hover:border-white/20'
                                                    }`}
                                                >
                                                    中文
                                                </button>
                                            </div>
                                        </div>

                                        {/* Storage Service Selector */}
                                        <div>
                                            <label className="flex items-center gap-2 text-xs font-medium text-white/80 mb-2">
                                                <HardDrive className="w-3.5 h-3.5 text-green-400" />
                                                {t.storage_service}
                                            </label>
                                            <div className="grid grid-cols-3 gap-2">
                                                {[
                                                    { id: 'off', label: t.storage_off },
                                                    { id: 's3', label: t.storage_s3 },
                                                    { id: 'webdav', label: t.storage_webdav }
                                                ].map(option => (
                                                    <button
                                                        key={option.id}
                                                        onClick={() => setStorageType(option.id as StorageType)}
                                                        className={`px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 border ${
                                                            storageType === option.id
                                                            ? 'bg-green-600/90 border-green-500/50 text-white shadow-lg shadow-green-900/20'
                                                            : 'bg-white/[0.03] border-white/10 text-white/60 hover:bg-white/[0.06] hover:text-white hover:border-white/20'
                                                        }`}
                                                    >
                                                        {option.label}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* HF Token */}
                                        {provider === 'huggingface' && (
                                            <div>
                                                <label className="flex items-center gap-2 text-xs font-medium text-white/80 mb-2">
                                                    <KeyRound className="w-3.5 h-3.5 text-yellow-500" />
                                                    {t.hfToken}
                                                </label>
                                                <div className="relative group">
                                                    <input
                                                        type={showToken ? "text" : "password"}
                                                        value={token}
                                                        onChange={handleTokenChange}
                                                        onPaste={handlePaste}
                                                        placeholder="hf_...,hf_..."
                                                        className="w-full pl-4 pr-10 py-2.5 bg-white/[0.03] border border-white/10 rounded-xl text-white placeholder:text-white/20 focus:outline-0 focus:ring-4 focus:ring-yellow-500/10 focus:border-yellow-500/50 hover:border-white/20 transition-all duration-300 ease-out font-mono text-xs"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() => setShowToken(!showToken)}
                                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-white/5"
                                                    >
                                                        {showToken ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                                                    </button>
                                                </div>
                                                
                                                {stats.total > 1 && (
                                                    <div className="mt-3 grid grid-cols-3 gap-2">
                                                        <div className="bg-white/[0.03] rounded-lg p-2.5 border border-white/[0.06] flex flex-col items-center group hover:border-white/10 transition-colors">
                                                            <span className="text-[10px] uppercase text-white/40 font-bold tracking-wider mb-0.5">{t.tokenTotal}</span>
                                                            <div className="flex items-center gap-1.5 text-white/90 font-mono text-xs font-medium">
                                                                <Database className="w-3 h-3 opacity-50 group-hover:opacity-100 transition-opacity" />
                                                                {stats.total}
                                                            </div>
                                                        </div>
                                                        <div className="bg-green-500/5 rounded-lg p-2.5 border border-green-500/10 flex flex-col items-center group hover:border-green-500/20 transition-colors">
                                                            <span className="text-[10px] uppercase text-green-400/60 font-bold tracking-wider mb-0.5">{t.tokenActive}</span>
                                                            <div className="flex items-center gap-1.5 text-green-400 font-mono text-xs font-medium">
                                                                <ShieldCheck className="w-3 h-3 opacity-70 group-hover:opacity-100 transition-opacity" />
                                                                {stats.active}
                                                            </div>
                                                        </div>
                                                        <div className="bg-red-500/5 rounded-lg p-2.5 border border-red-500/10 flex flex-col items-center group hover:border-red-500/20 transition-colors">
                                                            <span className="text-[10px] uppercase text-red-400/60 font-bold tracking-wider mb-0.5">{t.tokenExhausted}</span>
                                                            <div className="flex items-center gap-1.5 text-red-400 font-mono text-xs font-medium">
                                                                <ShieldAlert className="w-3 h-3 opacity-70 group-hover:opacity-100 transition-opacity" />
                                                                {stats.exhausted}
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}

                                                <p className="mt-2 text-[10px] text-white/40 leading-relaxed pl-1">
                                                    {t.hfTokenHelp} <a className="text-yellow-500 hover:text-yellow-400 underline decoration-yellow-500/30 underline-offset-2 transition-colors" href="https://huggingface.co/settings/tokens" target="_blank">{t.hfTokenLink}</a> {t.hfTokenHelpEnd}
                                                </p>
                                            </div>
                                        )}

                                        {/* Gitee Token */}
                                        {provider === 'gitee' && (
                                            <div>
                                                <label className="flex items-center gap-2 text-xs font-medium text-white/80 mb-2">
                                                    <KeyRound className="w-3.5 h-3.5 text-red-500" />
                                                    {t.giteeToken}
                                                </label>
                                                <div className="relative group">
                                                    <input
                                                        type={showGiteeToken ? "text" : "password"}
                                                        value={giteeToken}
                                                        onChange={handleGiteeTokenChange}
                                                        onPaste={handleGiteePaste}
                                                        placeholder="...,..."
                                                        className="w-full pl-4 pr-10 py-2.5 bg-white/[0.03] border border-white/10 rounded-xl text-white placeholder:text-white/20 focus:outline-0 focus:ring-4 focus:ring-red-500/10 focus:border-red-500/50 hover:border-white/20 transition-all duration-300 ease-out font-mono text-xs"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() => setShowGiteeToken(!showGiteeToken)}
                                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-white/5"
                                                    >
                                                        {showGiteeToken ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                                                    </button>
                                                </div>
                                                
                                                {giteeStats.total > 1 && (
                                                    <div className="mt-3 grid grid-cols-3 gap-2">
                                                        <div className="bg-white/[0.03] rounded-lg p-2.5 border border-white/[0.06] flex flex-col items-center">
                                                            <span className="text-[10px] uppercase text-white/40 font-bold tracking-wider">{t.tokenTotal}</span>
                                                            <div className="flex items-center gap-1.5 text-white/90 font-mono text-xs font-medium">
                                                                <Database className="w-3 h-3" />
                                                                {giteeStats.total}
                                                            </div>
                                                        </div>
                                                        <div className="bg-green-500/5 rounded-lg p-2.5 border border-green-500/10 flex flex-col items-center">
                                                            <span className="text-[10px] uppercase text-green-400/60 font-bold tracking-wider">{t.tokenActive}</span>
                                                            <div className="flex items-center gap-1.5 text-green-400 font-mono text-xs font-medium">
                                                                <ShieldCheck className="w-3 h-3" />
                                                                {giteeStats.active}
                                                            </div>
                                                        </div>
                                                        <div className="bg-red-500/5 rounded-lg p-2.5 border border-red-500/10 flex flex-col items-center">
                                                            <span className="text-[10px] uppercase text-red-400/60 font-bold tracking-wider">{t.tokenExhausted}</span>
                                                            <div className="flex items-center gap-1.5 text-red-400 font-mono text-xs font-medium">
                                                                <ShieldAlert className="w-3 h-3" />
                                                                {giteeStats.exhausted}
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}

                                                <p className="mt-2 text-[10px] text-white/40 leading-relaxed pl-1">
                                                    {t.giteeTokenHelp} <a className="text-red-500 hover:text-red-400 underline decoration-red-500/30 underline-offset-2 transition-colors" href="https://ai.gitee.com/dashboard/settings/tokens" target="_blank">{t.giteeTokenLink}</a> {t.giteeTokenHelpEnd}
                                                </p>
                                            </div>
                                        )}

                                        {/* Model Scope Token */}
                                        {provider === 'modelscope' && (
                                            <div>
                                                <label className="flex items-center gap-2 text-xs font-medium text-white/80 mb-2">
                                                    <KeyRound className="w-3.5 h-3.5 text-blue-500" />
                                                    {t.msToken}
                                                </label>
                                                <div className="relative group">
                                                    <input
                                                        type={showMsToken ? "text" : "password"}
                                                        value={msToken}
                                                        onChange={handleMsTokenChange}
                                                        onPaste={handleMsPaste}
                                                        placeholder="...,..."
                                                        className="w-full pl-4 pr-10 py-2.5 bg-white/[0.03] border border-white/10 rounded-xl text-white placeholder:text-white/20 focus:outline-0 focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500/50 hover:border-white/20 transition-all duration-300 ease-out font-mono text-xs"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() => setShowMsToken(!showMsToken)}
                                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-white/5"
                                                    >
                                                        {showMsToken ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                                                    </button>
                                                </div>
                                                
                                                {msStats.total > 1 && (
                                                    <div className="mt-3 grid grid-cols-3 gap-2">
                                                        <div className="bg-white/[0.03] rounded-lg p-2.5 border border-white/[0.06] flex flex-col items-center">
                                                            <span className="text-[10px] uppercase text-white/40 font-bold tracking-wider">{t.tokenTotal}</span>
                                                            <div className="flex items-center gap-1.5 text-white/90 font-mono text-xs font-medium">
                                                                <Database className="w-3 h-3" />
                                                                {msStats.total}
                                                            </div>
                                                        </div>
                                                        <div className="bg-green-500/5 rounded-lg p-2.5 border border-green-500/10 flex flex-col items-center">
                                                            <span className="text-[10px] uppercase text-green-400/60 font-bold tracking-wider">{t.tokenActive}</span>
                                                            <div className="flex items-center gap-1.5 text-green-400 font-mono text-xs font-medium">
                                                                <ShieldCheck className="w-3 h-3" />
                                                                {msStats.active}
                                                            </div>
                                                        </div>
                                                        <div className="bg-red-500/5 rounded-lg p-2.5 border border-red-500/10 flex flex-col items-center">
                                                            <span className="text-[10px] uppercase text-red-400/60 font-bold tracking-wider">{t.tokenExhausted}</span>
                                                            <div className="flex items-center gap-1.5 text-red-400 font-mono text-xs font-medium">
                                                                <ShieldAlert className="w-3 h-3" />
                                                                {msStats.exhausted}
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}

                                                <p className="mt-2 text-[10px] text-white/40 leading-relaxed pl-1">
                                                    {t.msTokenHelp} <a className="text-blue-500 hover:text-blue-400 underline decoration-blue-500/30 underline-offset-2 transition-colors" href="https://modelscope.cn/my/myaccesstoken" target="_blank">{t.msTokenLink}</a> {t.msTokenHelpEnd}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Tab 2: Prompt */}
                                {tab.id === 'prompt' && (
                                    <div className="space-y-7">
                                        {/* Optimization Model */}
                                        <div className="flex justify-between gap-2">
                                            <label className="flex items-center gap-2 text-sm font-medium text-white/80 mb-2.5">
                                                <Brain className="w-4 h-4 text-cyan-400" />
                                                {t.optimizationModel}
                                            </label>
                                            <div className="relative group w-2/3">
                                                <input 
                                                    type="text"
                                                    value={optimModel}
                                                    onChange={(e) => setOptimModel(e.target.value)}
                                                    placeholder={DEFAULT_OPTIMIZATION_MODELS[provider]}
                                                    className="w-full px-4 py-3 bg-white/[0.03] border border-white/10 rounded-xl text-white placeholder:text-white/20 focus:outline-0 focus:ring-4 focus:ring-cyan-500/10 focus:border-cyan-500/50 hover:border-white/20 transition-all duration-300 ease-out font-mono text-sm"
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-4">
                                            <div className="flex items-center justify-between">
                                                <label className="flex items-center gap-2 text-sm font-medium text-white/80">
                                                    <MessageSquare className="w-4 h-4 text-pink-400" />
                                                    {t.systemPrompts}
                                                </label>
                                                
                                                <button
                                                    onClick={handleRestoreDefault}
                                                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white/50 hover:text-white bg-white/5 hover:bg-white/10 transition-colors border border-transparent hover:border-white/10"
                                                    title={t.restoreDefault}
                                                >
                                                    <RotateCcw className="w-3.5 h-3.5" />
                                                    {t.restoreDefault}
                                                </button>
                                            </div>

                                            <div className="relative group">
                                                <textarea 
                                                    value={systemPrompt}
                                                    onChange={(e) => setSystemPrompt(e.target.value)}
                                                    placeholder={t.promptContent}
                                                    className="w-full h-32 bg-white/[0.03] border border-white/10 rounded-xl p-4 text-sm text-white/80 placeholder:text-white/20 focus:outline-0 focus:ring-4 focus:ring-pink-500/10 focus:border-pink-500/50 hover:border-white/20 resize-none custom-scrollbar leading-relaxed font-mono transition-all duration-300 ease-out"
                                                />
                                            </div>

                                            <p className="mt-1 text-xs text-white/40 leading-relaxed pl-1">
                                                {t.systemPromptHelp}
                                            </p>
                                        </div>
                                    </div>
                                )}

                                {/* Tab 3: Live Settings */}
                                {tab.id === 'live' && (
                                    <div className="space-y-6">
                                        {provider === 'modelscope' ? (
                                            <div className="flex flex-col items-center justify-center h-full text-white/40 space-y-4 pt-10">
                                                <Film className="w-12 h-12 opacity-50" />
                                                <p>{t.liveNotSupported}</p>
                                            </div>
                                        ) : (
                                            <div className="space-y-6">
                                                {/* Video Prompt */}
                                                <div className="space-y-4">
                                                    <div className="flex items-center justify-between">
                                                        <label className="flex items-center gap-2 text-sm font-medium text-white/80">
                                                            <MessageSquare className="w-4 h-4 text-purple-400" />
                                                            {t.videoPrompt}
                                                        </label>
                                                        <button
                                                            onClick={handleRestoreVideoDefaults}
                                                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white/50 hover:text-white bg-white/5 hover:bg-white/10 transition-colors border border-transparent hover:border-white/10"
                                                        >
                                                            <RotateCcw className="w-3.5 h-3.5" />
                                                            {t.restoreDefault}
                                                        </button>
                                                    </div>
                                                    <textarea 
                                                        value={videoSettings.prompt}
                                                        onChange={(e) => setVideoSettings({ ...videoSettings, prompt: e.target.value })}
                                                        className="w-full h-24 bg-white/[0.03] border border-white/10 rounded-xl p-4 text-sm text-white/90 placeholder:text-white/20 focus:outline-0 focus:ring-4 focus:ring-purple-500/10 focus:border-purple-500/50 hover:border-white/20 resize-none custom-scrollbar leading-relaxed font-mono transition-all duration-300 ease-out"
                                                    />
                                                </div>

                                                <div className="space-y-6">
                                                    {/* Duration */}
                                                    <div className="flex items-center justify-between gap-4">
                                                        <label className="flex items-center gap-2 text-sm font-medium text-white/80 min-w-[6rem]">
                                                            <Clock className="w-4 h-4 text-blue-400" />
                                                            {t.videoDuration}
                                                        </label>
                                                        <div className="flex flex-1 items-center gap-3">
                                                            <input
                                                                type="range"
                                                                min="0.5"
                                                                max="5"
                                                                step="0.5"
                                                                value={videoSettings.duration}
                                                                onChange={(e) => setVideoSettings({ ...videoSettings, duration: Number(e.target.value) })}
                                                                className="custom-range text-blue-500 flex-1"
                                                            />
                                                            <span className="text-xs font-mono text-white/50 bg-white/5 px-2 py-0.5 rounded min-w-[3.5rem] text-center">
                                                                {videoSettings.duration} {t.seconds}
                                                            </span>
                                                        </div>
                                                    </div>

                                                    {/* Steps */}
                                                    <div className="flex items-center justify-between gap-4">
                                                        <label className="flex items-center gap-2 text-sm font-medium text-white/80 min-w-[6rem]">
                                                            <Layers className="w-4 h-4 text-green-400" />
                                                            {t.videoSteps}
                                                        </label>
                                                        <div className="flex flex-1 items-center gap-3">
                                                            <input
                                                                type="range"
                                                                min="1"
                                                                max="30"
                                                                step="1"
                                                                value={videoSettings.steps}
                                                                onChange={(e) => setVideoSettings({ ...videoSettings, steps: Number(e.target.value) })}
                                                                className="custom-range text-green-500 flex-1"
                                                            />
                                                            <span className="text-xs font-mono text-white/50 bg-white/5 px-2 py-0.5 rounded min-w-[2rem] text-center">
                                                                {videoSettings.steps}
                                                            </span>
                                                        </div>
                                                    </div>

                                                    {/* Guidance */}
                                                    <div className="flex items-center justify-between gap-4">
                                                        <label className="flex items-center gap-2 text-sm font-medium text-white/80 min-w-[6rem]">
                                                            <Sparkles className="w-4 h-4 text-yellow-400" />
                                                            {t.videoGuidance}
                                                        </label>
                                                        <div className="flex flex-1 items-center gap-3">
                                                            <input
                                                                type="range"
                                                                min="0"
                                                                max="10"
                                                                step="1"
                                                                value={videoSettings.guidance}
                                                                onChange={(e) => setVideoSettings({ ...videoSettings, guidance: Number(e.target.value) })}
                                                                className="custom-range text-yellow-500 flex-1"
                                                            />
                                                            <span className="text-xs font-mono text-white/50 bg-white/5 px-2 py-0.5 rounded min-w-[2rem] text-center">
                                                                {videoSettings.guidance}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Tab 4: Cloud Storage (S3) */}
                                {tab.id === 's3' && (
                                     <div className="space-y-6">
                                        <div className="space-y-4">
                                            {/* Access Key */}
                                            <div className="flex items-center justify-between gap-4">
                                                <label className="text-sm font-medium text-white/80 w-1/3 flex-shrink-0">{t.s3_access_key}</label>
                                                <input 
                                                    type="text"
                                                    value={s3Config.accessKeyId}
                                                    onChange={(e) => setS3Config({ ...s3Config, accessKeyId: e.target.value })}
                                                    className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-green-500/50 transition-all font-mono"
                                                />
                                            </div>

                                            {/* Secret Key */}
                                            <div className="flex items-center justify-between gap-4">
                                                <label className="text-sm font-medium text-white/80 w-1/3 flex-shrink-0">{t.s3_secret_key}</label>
                                                <div className="relative w-full">
                                                    <input 
                                                        type={showS3Secret ? "text" : "password"}
                                                        value={s3Config.secretAccessKey}
                                                        onChange={(e) => setS3Config({ ...s3Config, secretAccessKey: e.target.value })}
                                                        className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-green-500/50 transition-all font-mono pr-8"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() => setShowS3Secret(!showS3Secret)}
                                                        className="absolute right-2 top-1/2 -translate-y-1/2 text-white/30 hover:text-white"
                                                    >
                                                        {showS3Secret ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Bucket & Region */}
                                            <div className="flex gap-4">
                                                <div className="flex-1 space-y-1">
                                                    <label className="text-xs font-medium text-white/60 block">{t.s3_bucket}</label>
                                                    <input 
                                                        type="text"
                                                        value={s3Config.bucket || ''}
                                                        onChange={(e) => setS3Config({ ...s3Config, bucket: e.target.value })}
                                                        className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-green-500/50 transition-all"
                                                    />
                                                </div>
                                                <div className="w-1/3 space-y-1">
                                                    <label className="text-xs font-medium text-white/60 block">{t.s3_region}</label>
                                                    <input 
                                                        type="text"
                                                        value={s3Config.region || ''}
                                                        onChange={(e) => setS3Config({ ...s3Config, region: e.target.value })}
                                                        className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-green-500/50 transition-all"
                                                    />
                                                </div>
                                            </div>

                                            {/* Endpoint */}
                                            <div className="flex-1 space-y-1">
                                                <label className="text-xs font-medium text-white/60 block">{t.s3_endpoint}</label>
                                                <input 
                                                    type="text"
                                                    value={s3Config.endpoint || ''}
                                                    onChange={(e) => setS3Config({ ...s3Config, endpoint: e.target.value })}
                                                    placeholder={getEndpointPlaceholder()}
                                                    className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-green-500/50 transition-all"
                                                />
                                            </div>
                                            
                                             <div className="flex gap-4">
                                                {/* Domain */}
                                                <div className="flex-1 space-y-1">
                                                    <label className="text-xs font-medium text-white/60 block">{t.s3_domain}</label>
                                                    <input 
                                                        type="text"
                                                        value={s3Config.publicDomain || ''}
                                                        onChange={(e) => setS3Config({ ...s3Config, publicDomain: e.target.value })}
                                                        placeholder={t.s3_domain_placeholder}
                                                        className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-green-500/50 transition-all"
                                                    />
                                                </div>

                                                {/* File Prefix */}
                                                <div className="flex-1 space-y-1">
                                                    <label className="text-xs font-medium text-white/60 block">{t.s3_prefix}</label>
                                                    <input 
                                                        type="text"
                                                        value={s3Config.prefix ?? 'peinture/'}
                                                        onChange={(e) => setS3Config({ ...s3Config, prefix: e.target.value })}
                                                        placeholder={t.s3_prefix_placeholder}
                                                        className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-green-500/50 transition-all font-mono"
                                                    />
                                                </div>
                                            </div>

                                            {/* Test Connection Button */}
                                            <div className="flex flex-col gap-2">
                                                <div className="flex justify-end">
                                                    <button 
                                                        onClick={handleTestS3}
                                                        disabled={isTestingS3}
                                                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600/20 text-green-400 hover:bg-green-600/30 border border-green-500/30 transition-all text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                                                    >
                                                        {isTestingS3 ? (
                                                            <>
                                                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                                {t.testing}
                                                            </>
                                                        ) : (
                                                            <>
                                                                <PlugZap className="w-3.5 h-3.5" />
                                                                {t.test_connection}
                                                            </>
                                                        )}
                                                    </button>
                                                </div>
                                                {testS3Result && (
                                                    <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 ${testS3Result.success ? 'bg-green-500/10 border-green-500/20 text-green-400' : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
                                                        {testS3Result.success ? <Check className="w-4 h-4 mt-0.5 flex-shrink-0" /> : <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />}
                                                        <span>{testS3Result.message}</span>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Tab 5: WebDAV Storage */}
                                {tab.id === 'webdav' && (
                                     <div className="space-y-6">
                                        <div className="space-y-4">
                                            {/* URL */}
                                            <div className="flex items-center justify-between gap-4">
                                                <label className="text-sm font-medium text-white/80 w-1/3 flex-shrink-0">{t.webdav_url}</label>
                                                <input 
                                                    type="text"
                                                    value={webdavConfig.url}
                                                    onChange={(e) => setWebdavConfig({ ...webdavConfig, url: e.target.value })}
                                                    placeholder={t.webdav_url_placeholder}
                                                    className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-blue-500/50 transition-all font-mono"
                                                />
                                            </div>

                                            {/* Username */}
                                            <div className="flex items-center justify-between gap-4">
                                                <label className="text-sm font-medium text-white/80 w-1/3 flex-shrink-0">{t.webdav_username}</label>
                                                <input 
                                                    type="text"
                                                    value={webdavConfig.username}
                                                    onChange={(e) => setWebdavConfig({ ...webdavConfig, username: e.target.value })}
                                                    className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-blue-500/50 transition-all font-mono"
                                                />
                                            </div>

                                            {/* Password */}
                                            <div className="flex items-center justify-between gap-4">
                                                <label className="text-sm font-medium text-white/80 w-1/3 flex-shrink-0">{t.webdav_password}</label>
                                                <div className="relative w-full">
                                                    <input 
                                                        type={showWebdavPass ? "text" : "password"}
                                                        value={webdavConfig.password}
                                                        onChange={(e) => setWebdavConfig({ ...webdavConfig, password: e.target.value })}
                                                        className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-blue-500/50 transition-all font-mono pr-8"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() => setShowWebdavPass(!showWebdavPass)}
                                                        className="absolute right-2 top-1/2 -translate-y-1/2 text-white/30 hover:text-white"
                                                    >
                                                        {showWebdavPass ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Directory */}
                                            <div className="flex items-center justify-between gap-4">
                                                <label className="text-sm font-medium text-white/80 w-1/3 flex-shrink-0">{t.webdav_directory}</label>
                                                <input 
                                                    type="text"
                                                    value={webdavConfig.directory}
                                                    onChange={(e) => setWebdavConfig({ ...webdavConfig, directory: e.target.value })}
                                                    className="w-full px-3 py-2 bg-white/[0.03] border border-white/10 rounded-lg text-white text-sm focus:outline-0 focus:border-blue-500/50 transition-all font-mono"
                                                />
                                            </div>

                                            {/* Test Connection Button */}
                                            <div className="flex flex-col gap-2">
                                                <div className="flex justify-end">
                                                    <button 
                                                        onClick={handleTestWebDAV}
                                                        disabled={isTestingWebDAV}
                                                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 border border-blue-500/30 transition-all text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                                                    >
                                                        {isTestingWebDAV ? (
                                                            <>
                                                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                                {t.testing}
                                                            </>
                                                        ) : (
                                                            <>
                                                                <PlugZap className="w-3.5 h-3.5" />
                                                                {t.test_connection}
                                                            </>
                                                        )}
                                                    </button>
                                                </div>
                                                {testWebDAVResult && (
                                                    <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 ${testWebDAVResult.success ? 'bg-green-500/10 border-green-500/20 text-green-400' : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
                                                        {testWebDAVResult.success ? <Check className="w-4 h-4 mt-0.5 flex-shrink-0" /> : <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />}
                                                        <span>{testWebDAVResult.message}</span>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="flex items-center justify-end gap-3 px-5 py-2 border-t border-white/[0.06] bg-white/[0.02] flex-shrink-0">
                    <button
                        onClick={onClose}
                        className="px-5 py-2.5 text-sm font-medium text-white/60 hover:text-white hover:bg-white/[0.06] rounded-lg transition-all duration-200"
                    >
                        {t.cancel}
                    </button>
                    <button
                        onClick={handleSave}
                        className="flex items-center gap-2 px-6 py-2.5 text-sm font-bold text-white bg-purple-600 hover:bg-purple-500 active:bg-purple-700 active:scale-95 rounded-lg transition-all shadow-[0_4px_20px_-4px_rgba(147,51,234,0.5)] hover:shadow-[0_4px_25px_-4px_rgba(147,51,234,0.6)]"
                    >
                        <Save className="w-4 h-4" />
                        {t.save}
                    </button>
                </div>
            </div>
        </div>
    );
};

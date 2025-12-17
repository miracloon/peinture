
import React, { useState } from 'react';
import { Select } from './Select';
import { Tooltip } from './Tooltip';
import { Settings, ChevronUp, ChevronDown, Minus, Plus, Dices, Cpu, Server } from 'lucide-react';
import { ModelOption, ProviderOption, AspectRatioOption } from '../types';
import { PROVIDER_OPTIONS, HF_MODEL_OPTIONS, GITEE_MODEL_OPTIONS, MS_MODEL_OPTIONS, Z_IMAGE_MODELS, FLUX_MODELS, getModelConfig, getGuidanceScaleConfig } from '../constants';

interface ControlPanelProps {
    provider: ProviderOption;
    setProvider: (val: ProviderOption) => void;
    model: ModelOption;
    setModel: (val: ModelOption) => void;
    aspectRatio: AspectRatioOption;
    setAspectRatio: (val: AspectRatioOption) => void;
    steps: number;
    setSteps: (val: number) => void;
    guidanceScale: number;
    setGuidanceScale: (val: number) => void;
    seed: string;
    setSeed: (val: string) => void;
    enableHD: boolean;
    setEnableHD: (val: boolean) => void;
    t: any;
    aspectRatioOptions: { value: string; label: string }[];
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
    provider,
    setProvider,
    model,
    setModel,
    aspectRatio,
    setAspectRatio,
    steps,
    setSteps,
    guidanceScale,
    setGuidanceScale,
    seed,
    setSeed,
    enableHD,
    setEnableHD,
    t,
    aspectRatioOptions
}) => {
    const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

    // Derived helpers
    const currentModelOptions = provider === 'gitee' ? GITEE_MODEL_OPTIONS : (provider === 'modelscope' ? MS_MODEL_OPTIONS : HF_MODEL_OPTIONS);
    const currentModelConfig = getModelConfig(provider, model);
    const guidanceScaleConfig = getGuidanceScaleConfig(model, provider);

    const handleRandomizeSeed = () => {
        setSeed(Math.floor(Math.random() * 2147483647).toString());
    };

    const handleAdjustSeed = (amount: number) => {
        const current = parseInt(seed || '0', 10);
        if (isNaN(current)) {
            setSeed((0 + amount).toString());
        } else {
            setSeed((current + amount).toString());
        }
    };

    // Handle Provider Change wrapper to reset model
    const onProviderChange = (newProvider: string) => {
        const p = newProvider as ProviderOption;
        setProvider(p);
        // Reset model to first option of the new provider to avoid mismatch
        if (p === 'gitee') {
            setModel(GITEE_MODEL_OPTIONS[0].value as ModelOption);
        } else if (p === 'modelscope') {
            setModel(MS_MODEL_OPTIONS[0].value as ModelOption);
        } else {
            setModel(HF_MODEL_OPTIONS[0].value as ModelOption);
        }
    };

    return (
        <div className="space-y-4 md:space-y-6">
            {/* Provider Selection */}
            <Select
                label={t.provider}
                value={provider}
                onChange={onProviderChange}
                options={PROVIDER_OPTIONS}
                icon={<Server className="w-5 h-5" />}
            />

            {/* Model Selection */}
            <Select
                label={t.model}
                value={model}
                onChange={(val) => setModel(val as ModelOption)}
                options={currentModelOptions}
                icon={<Cpu className="w-5 h-5" />}
                headerContent={
                    (Z_IMAGE_MODELS.includes(model) || FLUX_MODELS.includes(model)) && (
                        <div className="flex items-center gap-2 animate-in fade-in duration-300">
                            <span className="text-xs font-medium text-white/50">{t.hd}</span>
                            <Tooltip content={enableHD ? t.hdEnabled : t.hdDisabled}>
                                <button
                                    onClick={() => setEnableHD(!enableHD)}
                                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500/50 ${enableHD ? 'bg-purple-600' : 'bg-white/10'}`}
                                >
                                    <span
                                        className={`${enableHD ? 'translate-x-4' : 'translate-x-1'} inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform`}
                                    />
                                </button>
                            </Tooltip>
                        </div>
                    )
                }
            />

            {/* Aspect Ratio */}
            <Select
                label={t.aspectRatio}
                value={aspectRatio}
                onChange={(val) => setAspectRatio(val as AspectRatioOption)}
                options={aspectRatioOptions}
            />

            {/* Advanced Settings */}
            <div className="border-t border-white/5 pt-4">
                <button
                    type="button"
                    onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                    className="flex items-center justify-between w-full text-left text-white/60 hover:text-purple-400 transition-colors group"
                >
                    <span className="text-sm font-medium flex items-center gap-2">
                        <Settings className="w-4 h-4 group-hover:rotate-45 transition-transform duration-300" />
                        {t.advancedSettings}
                    </span>
                    {isAdvancedOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                <div className={`grid transition-[grid-template-rows] duration-300 ease-in-out ${isAdvancedOpen ? 'grid-rows-[1fr] mt-4' : 'grid-rows-[0fr]'}`}>
                    <div className="overflow-hidden">
                        <div className="space-y-5">
                            {/* Steps */}
                            <div className="group">
                                <div className="flex items-center justify-between pb-2">
                                    <p className="text-white/80 text-sm font-medium">{t.steps}</p>
                                    <span className="text-white/50 text-xs bg-white/5 px-2 py-0.5 rounded font-mono">{steps}</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <input
                                        type="range"
                                        min={currentModelConfig.min}
                                        max={currentModelConfig.max}
                                        value={steps}
                                        onChange={(e) => setSteps(Number(e.target.value))}
                                        className="custom-range text-purple-500"
                                    />
                                </div>
                            </div>

                            {/* Guidance Scale - Only for Flux Models */}
                            {guidanceScaleConfig && (
                                <div className="group">
                                    <div className="flex items-center justify-between pb-2">
                                        <p className="text-white/80 text-sm font-medium">{t.guidanceScale}</p>
                                        <span className="text-white/50 text-xs bg-white/5 px-2 py-0.5 rounded font-mono">{guidanceScale.toFixed(1)}</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <input
                                            type="range"
                                            min={guidanceScaleConfig.min}
                                            max={guidanceScaleConfig.max}
                                            step={guidanceScaleConfig.step}
                                            value={guidanceScale}
                                            onChange={(e) => setGuidanceScale(Number(e.target.value))}
                                            className="custom-range text-purple-500"
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Seed */}
                            <div className="group">
                                <div className="flex items-center justify-between pb-2">
                                    <p className="text-white/80 text-sm font-medium">{t.seed}</p>
                                    <span className="text-white/40 text-xs">{t.seedOptional}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="flex flex-1 items-center rounded-lg border border-white/10 bg-white/5 focus-within:ring-2 focus-within:ring-purple-500/50 focus-within:border-purple-500 transition-all h-10 overflow-hidden">
                                        <button
                                            onClick={() => handleAdjustSeed(-1)}
                                            className="h-full px-2 text-white/40 hover:text-white hover:bg-white/5 transition-colors border-r border-white/5"
                                        >
                                            <Minus className="w-3.5 h-3.5" />
                                        </button>
                                        <input
                                            type="number"
                                            value={seed}
                                            onChange={(e) => setSeed(e.target.value)}
                                            className="form-input flex-1 h-full bg-transparent border-none text-white/90 focus:ring-0 placeholder:text-white/30 px-2 text-xs font-mono text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                                            placeholder={t.seedPlaceholder}
                                        />
                                        <button
                                            onClick={() => handleAdjustSeed(1)}
                                            className="h-full px-2 text-white/40 hover:text-white hover:bg-white/5 transition-colors border-l border-white/5"
                                        >
                                            <Plus className="w-3.5 h-3.5" />
                                        </button>
                                    </div>

                                    <Tooltip content={t.seedPlaceholder}>
                                        <button
                                            onClick={handleRandomizeSeed}
                                            className="flex h-10 w-10 shrink-0 cursor-pointer items-center justify-center overflow-hidden rounded-lg bg-white/10 text-white/60 hover:bg-white/20 hover:text-white transition-colors active:scale-95"
                                        >
                                            <Dices className="w-4 h-4" />
                                        </button>
                                    </Tooltip>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

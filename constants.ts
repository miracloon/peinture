
import { ModelOption, ProviderOption } from './types';

export const HF_MODEL_OPTIONS = [
  { value: 'z-image-turbo', label: 'Z-Image Turbo' },
  { value: 'qwen-image-fast', label: 'Qwen Image Fast' },
  { value: 'ovis-image', label: 'Ovis Image' },
  { value: 'flux-1-schnell', label: 'FLUX.1 Schnell' }
];

export const GITEE_MODEL_OPTIONS = [
  { value: 'z-image-turbo', label: 'Z-Image Turbo' },
  { value: 'Qwen-Image', label: 'Qwen Image' },
  { value: 'flux-1-schnell', label: 'FLUX.1 Schnell' },
  { value: 'FLUX_1-Krea-dev', label: 'FLUX.1 Krea' },
  { value: 'FLUX.1-dev', label: 'FLUX.1' }
];

export const MS_MODEL_OPTIONS = [
  { value: 'Tongyi-MAI/Z-Image-Turbo', label: 'Z-Image Turbo' },
  { value: 'black-forest-labs/FLUX.2-dev', label: 'FLUX.2' },
  { value: 'black-forest-labs/FLUX.1-Krea-dev', label: 'FLUX.1 Krea' },
  { value: 'MusePublic/489_ckpt_FLUX_1', label: 'FLUX.1' }
];

export const PROVIDER_OPTIONS = [
    { value: 'huggingface', label: 'Hugging Face' },
    { value: 'gitee', label: 'Gitee AI' },
    { value: 'modelscope', label: 'Model Scope' }
];

export const FLUX_MODELS = [
    'flux-1-schnell', 
    'FLUX_1-Krea-dev', 
    'FLUX.1-dev',
    'black-forest-labs/FLUX.2-dev',
    'black-forest-labs/FLUX.1-Krea-dev',
    'MusePublic/489_ckpt_FLUX_1'
];

export const Z_IMAGE_MODELS = ['z-image-turbo', 'Tongyi-MAI/Z-Image-Turbo'];

export const getModelConfig = (provider: ProviderOption, model: ModelOption) => {
  if (provider === 'gitee') {
    if (model === 'z-image-turbo') return { min: 1, max: 20, default: 9 };
    if (model === 'Qwen-Image') return { min: 4, max: 50, default: 20 };
    if (model === 'flux-1-schnell') return { min: 1, max: 50, default: 8 };
    if (model === 'FLUX_1-Krea-dev') return { min: 1, max: 50, default: 20 };
    if (model === 'FLUX.1-dev') return { min: 1, max: 50, default: 20 };
  } else if (provider === 'modelscope') {
    if (model === 'Tongyi-MAI/Z-Image-Turbo') return { min: 1, max: 20, default: 9 };
    if (model === 'black-forest-labs/FLUX.2-dev') return { min: 1, max: 50, default: 24 };
    if (model === 'black-forest-labs/FLUX.1-Krea-dev') return { min: 1, max: 50, default: 24 };
    if (model === 'MusePublic/489_ckpt_FLUX_1') return { min: 1, max: 50, default: 24 };
  } else {
    if (model === 'z-image-turbo') return { min: 1, max: 20, default: 9 };
    if (model === 'flux-1-schnell') return { min: 1, max: 50, default: 8 };
    if (model === 'qwen-image-fast') return { min: 4, max: 28, default: 8 };
    if (model === 'ovis-image') return { min: 1, max: 50, default: 24 };
  }
  return { min: 1, max: 20, default: 9 }; // fallback
};

export const getGuidanceScaleConfig = (model: ModelOption, provider: ProviderOption) => {
  if (provider === 'gitee') {
    if (model === 'flux-1-schnell') return { min: 0, max: 50, step: 0.1, default: 7.5 };
    if (model === 'FLUX_1-Krea-dev') return { min: 0, max: 20, step: 0.1, default: 4.5 };
    if (model === 'FLUX.1-dev') return { min: 0, max: 20, step: 0.1, default: 4.5 };
  } else if (provider === 'modelscope') {
    if (model === 'black-forest-labs/FLUX.2-dev') return { min: 1, max: 10, step: 0.1, default: 3.5 };
    if (model === 'black-forest-labs/FLUX.1-Krea-dev') return { min: 1, max: 20, step: 0.1, default: 3.5 };
    if (model === 'MusePublic/489_ckpt_FLUX_1') return { min: 1, max: 20, step: 0.1, default: 3.5 };
  }
  return null;
};

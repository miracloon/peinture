
export interface GeneratedImage {
    id: string;
    url: string;
    prompt: string;
    aspectRatio: string;
    timestamp: number;
    model: string;
    seed?: number;
    steps?: number;
    guidanceScale?: number;
    duration?: number;
    isBlurred?: boolean;
    isUpscaled?: boolean;
    provider?: ProviderOption;
    // Video Generation Properties
    videoUrl?: string;
    videoTaskId?: string;
    videoStatus?: 'generating' | 'success' | 'failed';
    videoError?: string;
    videoProvider?: ProviderOption;
}

export type AspectRatioOption = "1:1" | "3:2" | "2:3" | "3:4" | "4:3" | "4:5" | "5:4" | "9:16" | "16:9";

export type ModelOption = 
    | "z-image-turbo" 
    | "qwen-image-fast" 
    | "ovis-image" 
    | "Qwen-Image"
    | "flux-1-schnell"
    | "FLUX_1-Krea-dev"
    | "FLUX.1-dev"
    | "FLUX.2-dev"
    | "Tongyi-MAI/Z-Image-Turbo"
    | "Qwen/Qwen-Image"
    | "black-forest-labs/FLUX.2-dev"
    | "black-forest-labs/FLUX.1-Krea-dev"
    | "MusePublic/489_ckpt_FLUX_1";

export type ProviderOption = "huggingface" | "gitee" | "modelscope";

export interface GenerationParams {
    model: ModelOption;
    prompt: string;
    aspectRatio: AspectRatioOption;
    seed?: number;
    steps?: number;
    guidanceScale?: number;
}

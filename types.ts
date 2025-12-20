
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

export interface CloudImage {
    id: string;
    url: string; // Cloud URL
    thumbnailUrl?: string;
    prompt: string;
    timestamp: number;
    fileName: string;
}

export interface CloudFile {
    key: string;
    lastModified: Date;
    size: number;
    url: string;
    type: 'image' | 'video' | 'unknown';
}

// Deprecated: Alias for backward compatibility if needed, but CloudFile is preferred
export type S3Object = CloudFile;

export type StorageType = 'off' | 's3' | 'webdav';

export interface S3Config {
    accessKeyId: string;
    secretAccessKey: string;
    bucket?: string; // Optional
    region?: string; // Optional
    endpoint?: string; // Optional custom endpoint
    publicDomain?: string; // Optional CDN/Public domain
    prefix?: string; // Optional prefix, default 'peinture/'
}

export interface WebDAVConfig {
    url: string;
    username: string;
    password: string;
    directory: string;
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

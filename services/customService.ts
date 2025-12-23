
import { CustomProvider, GeneratedImage, AspectRatioOption, RemoteModelList } from "../types";
import { generateUUID } from "./utils";

const cleanUrl = (url: string) => url.replace(/\/+$/, '');

const handleResponse = async (response: Response) => {
    if (!response.ok) {
        // Specifically throw error with status for 401 handling
        if (response.status === 401) {
            throw new Error(`401`);
        }
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
    }
    const data = await response.json();
    
    // Support various common response formats for flexibility
    if (typeof data === 'string') return data; // Raw string response
    if (data.url) return data.url[0]; // Standard { id: "", url: ["..."] }
    
    // For text responses
    if (data.text) return data.text;
    if (data.content) return data.content;
    if (data.choices && data.choices[0]?.message?.content) return data.choices[0].message.content;
    
    // Fallback: return the whole object if we can't parse a specific field, caller might handle it
    return data;
};

export const fetchServerModels = async (token?: string): Promise<RemoteModelList> => {
    const headers: Record<string, string> = {};
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    const response = await fetch('/api/v1/models', { headers });
    if (!response.ok) {
        if (response.status === 401) throw new Error('401');
        throw new Error('Failed to fetch server models');
    }
    return await response.json();
};

export const generateCustomImage = async (
    provider: CustomProvider,
    model: string,
    prompt: string,
    aspectRatio: AspectRatioOption,
    seed?: number,
    steps?: number,
    guidance?: number,
    enableHD?: boolean
): Promise<GeneratedImage> => {
    const baseUrl = cleanUrl(provider.apiUrl);
    const body = {
        model,
        prompt,
        ar: aspectRatio,
        seed: seed ?? Math.floor(Math.random() * 2147483647),
        steps,
        guidance,
        enableHD
    };

    const response = await fetch(`${baseUrl}/v1/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': provider.token ? `Bearer ${provider.token}` : ''
        },
        body: JSON.stringify(body)
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
    }
    const { id, url } = await response.json();

    if (typeof url !== 'string') {
        throw new Error("Invalid response format from custom provider: URL not found");
    }

    return {
        id: id || generateUUID(),
        url,
        model,
        prompt,
        aspectRatio,
        timestamp: Date.now(),
        seed: body.seed,
        steps,
        provider: provider.id // Use custom provider ID
    };
};

export const editImageCustom = async (
    provider: CustomProvider,
    model: string,
    imageBlobs: (string | Blob | File)[],
    prompt: string,
    seed?: number,
    steps?: number,
    guidance?: number
): Promise<GeneratedImage> => {
    const baseUrl = cleanUrl(provider.apiUrl);
    const formData = new FormData();
    formData.append('model', model);
    formData.append('prompt', prompt);
    if (seed !== undefined) formData.append('seed', seed.toString());
    if (steps !== undefined) formData.append('steps', steps.toString());
    if (guidance !== undefined) formData.append('guidance', guidance.toString());
    
    imageBlobs.forEach((blob) => {
        formData.append('image', blob);
    });

    const headers: Record<string, string> = {};
    if (provider.token) headers['Authorization'] = `Bearer ${provider.token}`;

    const response = await fetch(`${baseUrl}/v1/edit`, {
        method: 'POST',
        headers, // Do not set Content-Type, let browser set it with boundary
        body: formData
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
    }
    const { id, url } = await response.json();

    if (typeof url !== 'string') {
        throw new Error("Invalid response format from custom provider: URL not found");
    }

    return {
        id: id || generateUUID(),
        url,
        model,
        prompt,
        aspectRatio: 'custom',
        timestamp: Date.now(),
        seed,
        steps,
        provider: provider.id
    };
};

export const generateCustomVideo = async (
    provider: CustomProvider,
    model: string,
    imageUrl: string,
    prompt: string,
    duration: number,
    seed: number,
    steps: number,
    guidance: number
): Promise<string> => {
    const baseUrl = cleanUrl(provider.apiUrl);
    const body = {
        model,
        imageUrl,
        prompt,
        duration,
        seed,
        steps,
        guidance
    };

    const result = await handleResponse(await fetch(`${baseUrl}/v1/video`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': provider.token ? `Bearer ${provider.token}` : ''
        },
        body: JSON.stringify(body)
    }));
    
    // Check if result is string URL or object
    const videoUrl = typeof result === 'string' ? result : result?.url[0];
    
    if (!videoUrl) throw new Error("Video URL not found in response");
    
    return videoUrl;
};

export const optimizePromptCustom = async (
    provider: CustomProvider,
    model: string,
    prompt: string
): Promise<string> => {
    const baseUrl = cleanUrl(provider.apiUrl);
    const body = {
        model,
        prompt
    };

    const response = await fetch(`${baseUrl}/v1/text`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': provider.token ? `Bearer ${provider.token}` : ''
        },
        body: JSON.stringify(body)
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
    }
    const { text } = await response.json();
    return text;
};

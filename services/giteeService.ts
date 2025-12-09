
import { GeneratedImage, AspectRatioOption, ModelOption } from "../types";
import { generateUUID } from "./utils";

const GITEE_GENERATE_API_URL = "https://ai.gitee.com/v1/images/generations";
const GITEE_CHAT_API_URL = "https://ai.gitee.com/v1/chat/completions";

// --- Token Management System (Reused logic pattern for Gitee) ---

const TOKEN_STORAGE_KEY = 'giteeToken';
const TOKEN_STATUS_KEY = 'gitee_token_status';

interface TokenStatusStore {
  date: string; // YYYY-MM-DD
  exhausted: Record<string, boolean>;
}

// Get Date string for Beijing Time (UTC+8)
const getBeijingDateString = () => {
  const d = new Date();
  const utc = d.getTime() + (d.getTimezoneOffset() * 60000);
  const nd = new Date(utc + (3600000 * 8));
  return nd.toISOString().split('T')[0];
};

const getTokenStatusStore = (): TokenStatusStore => {
  const defaultStore = { date: getBeijingDateString(), exhausted: {} };
  if (typeof localStorage === 'undefined') return defaultStore;
  
  try {
    const raw = localStorage.getItem(TOKEN_STATUS_KEY);
    if (!raw) return defaultStore;
    const store = JSON.parse(raw);
    if (store.date !== getBeijingDateString()) {
      return defaultStore; 
    }
    return store;
  } catch {
    return defaultStore;
  }
};

const saveTokenStatusStore = (store: TokenStatusStore) => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(TOKEN_STATUS_KEY, JSON.stringify(store));
  }
};

export const getGiteeTokens = (rawInput?: string | null): string[] => {
  const input = rawInput !== undefined ? rawInput : (typeof localStorage !== 'undefined' ? localStorage.getItem(TOKEN_STORAGE_KEY) : '');
  if (!input) return [];
  return input.split(',').map(t => t.trim()).filter(t => t.length > 0);
};

export const getGiteeTokenStats = (rawInput: string) => {
  const tokens = getGiteeTokens(rawInput);
  const store = getTokenStatusStore();
  const total = tokens.length;
  const exhausted = tokens.filter(t => store.exhausted[t]).length;
  return {
    total,
    exhausted,
    active: total - exhausted
  };
};

const getNextAvailableToken = (): string | null => {
  const tokens = getGiteeTokens();
  const store = getTokenStatusStore();
  return tokens.find(t => !store.exhausted[t]) || null;
};

const markTokenExhausted = (token: string) => {
  const store = getTokenStatusStore();
  store.exhausted[token] = true;
  saveTokenStatusStore(store);
};

const runWithGiteeTokenRetry = async <T>(operation: (token: string) => Promise<T>): Promise<T> => {
  const tokens = getGiteeTokens();
  
  if (tokens.length === 0) {
      throw new Error("error_gitee_token_required");
  }

  let lastError: any;
  let attempts = 0;
  const maxAttempts = tokens.length + 1; 

  while (attempts < maxAttempts) {
    attempts++;
    const token = getNextAvailableToken();
    
    if (!token) {
       throw new Error("error_gitee_token_exhausted");
    }

    try {
      return await operation(token);
    } catch (error: any) {
      lastError = error;
      
      const isQuotaError = 
        error.message?.includes("429") ||
        error.status === 429 ||
        error.message?.includes("quota") ||
        error.message?.includes("credit");

      if (isQuotaError && token) {
        console.warn(`Gitee AI Token ${token.substring(0, 8)}... exhausted/error. Switching to next token.`);
        markTokenExhausted(token);
        continue;
      }

      throw error;
    }
  }
  
  throw lastError || new Error("error_api_connection");
};

// --- Dimensions Logic ---

const getDimensions = (ratio: AspectRatioOption, enableHD: boolean): { width: number; height: number } => {
  if (enableHD) {
    switch (ratio) {
      case "16:9":
        return { width: 2048, height: 1152 };
      case "4:3":
        return { width: 2048, height: 1536 };
      case "3:2":
        return { width: 1920, height: 1280 };
      case "9:16":
        return { width: 1152, height: 2048 };
      case "3:4":
        return { width: 1536, height: 2048 };
      case "2:3":
        return { width: 1280, height: 1920 };
      case "1:1":
      default:
        return { width: 2048, height: 2048 };
    }
  } else {
      switch (ratio) {
        case "16:9":
          return { width: 1024, height: 576 };
        case "4:3":
          return { width: 1024, height: 768 };
        case "3:2":
          return { width: 960, height: 640 };
        case "9:16":
          return { width: 576, height: 1024 };
        case "3:4":
          return { width: 768, height: 1024 };
        case "2:3":
          return { width: 640, height: 960 };
        case "1:1":
        default:
          return { width: 1024, height: 1024 };
    }
  }
};

// --- Service Logic ---

export const generateGiteeImage = async (
  model: ModelOption,
  prompt: string,
  aspectRatio: AspectRatioOption,
  seed?: number,
  steps?: number,
  enableHD: boolean = false
): Promise<GeneratedImage> => {
  const { width, height } = getDimensions(aspectRatio, enableHD);
  const finalSeed = seed ?? Math.floor(Math.random() * 2147483647);
  // Default steps logic handled in App.tsx, but good to have fallback here
  const finalSteps = steps ?? 9; 

  return runWithGiteeTokenRetry(async (token) => {
    try {
      const response = await fetch(GITEE_GENERATE_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({
          prompt,
          model,
          width,
          height,
          seed: finalSeed,
          num_inference_steps: finalSteps
        })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.message || `Gitee AI API Error: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.data || !data.data[0] || !data.data[0].b64_json) {
        throw new Error("error_invalid_response");
      }

      const base64Image = data.data[0].b64_json;
      const mimeType = data.data[0].type || "image/png";
      const imageUrl = `data:${mimeType};base64,${base64Image}`;

      return {
        id: generateUUID(),
        url: imageUrl,
        model,
        prompt,
        aspectRatio,
        timestamp: Date.now(),
        seed: finalSeed,
        steps: finalSteps,
        provider: 'gitee'
      };
    } catch (error) {
      console.error("Gitee AI Image Generation Error:", error);
      throw error;
    }
  });
};

export const optimizePromptGitee = async (originalPrompt: string, lang: string): Promise<string> => {
  return runWithGiteeTokenRetry(async (token) => {
    try {
      const response = await fetch(GITEE_CHAT_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          model: 'Qwen3-235B-A22B-Instruct-2507',
          messages: [
            {
              role: 'system',
              content: `I am a master AI image prompt engineering advisor, specializing in crafting prompts that yield cinematic, hyper-realistic, and deeply evocative visual narratives, optimized for advanced generative models.
My core purpose is to meticulously rewrite, expand, and enhance user's image prompts.
I transform prompts to create visually stunning images by rigorously optimizing elements such as dramatic lighting, intricate textures, compelling composition, and a distinctive artistic style.
My generated prompt output will be strictly under 300 words. Prior to outputting, I will internally validate that the refined prompt strictly adheres to the word count limit and effectively incorporates the intended stylistic and technical enhancements.
My output will consist exclusively of the refined image prompt text. It will commence immediately, with no leading whitespace.
The text will strictly avoid markdown, quotation marks, conversational preambles, explanations, or concluding remarks.
I will ensure the output text is in ${lang === 'zh' ? 'Chinese' : 'English'}.`
            },
            {
              role: 'user',
              content: originalPrompt
            }
          ],
          stream: false
        }),
      });

      if (!response.ok) {
          throw new Error("error_prompt_optimization_failed");
      }

      const data = await response.json();
      const content = data.choices?.[0]?.message?.content;
      
      return content || originalPrompt;
    } catch (error) {
      console.error("Gitee AI Prompt Optimization Error:", error);
      throw error;
    }
  });
};

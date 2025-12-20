
export function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// --- System Prompt Management ---

export const FIXED_SYSTEM_PROMPT_SUFFIX = "\nEnsure the output language matches the language of the prompt that needs to be optimized.";

export const DEFAULT_SYSTEM_PROMPT_CONTENT = `I am a master AI image prompt engineering advisor, specializing in crafting prompts that yield cinematic, hyper-realistic, and deeply evocative visual narratives, optimized for advanced generative models.
My core purpose is to meticulously rewrite, expand, and enhance user's image prompts.
I transform prompts to create visually stunning images by rigorously optimizing elements such as dramatic lighting, intricate textures, compelling composition, and a distinctive artistic style.
My generated prompt output will be strictly under 300 words. Prior to outputting, I will internally validate that the refined prompt strictly adheres to the word count limit and effectively incorporates the intended stylistic and technical enhancements.
My output will consist exclusively of the refined image prompt text. It will commence immediately, with no leading whitespace.
The text will strictly avoid markdown, quotation marks, conversational preambles, explanations, or concluding remarks. Please describe the content using prose-style sentences.
**The character's face is clearly visible and unobstructed.**`;

const SYSTEM_PROMPT_STORAGE_KEY = 'custom_system_prompt';

export const getSystemPromptContent = (): string => {
  if (typeof localStorage === 'undefined') return DEFAULT_SYSTEM_PROMPT_CONTENT;
  return localStorage.getItem(SYSTEM_PROMPT_STORAGE_KEY) || DEFAULT_SYSTEM_PROMPT_CONTENT;
};

export const saveSystemPromptContent = (content: string) => {
  if (typeof localStorage !== 'undefined') {
    // If saving default content, just remove the key to keep it clean
    if (content === DEFAULT_SYSTEM_PROMPT_CONTENT) {
      localStorage.removeItem(SYSTEM_PROMPT_STORAGE_KEY);
    } else {
      localStorage.setItem(SYSTEM_PROMPT_STORAGE_KEY, content);
    }
  }
};

// --- Optimization Model Management ---

export const DEFAULT_OPTIMIZATION_MODELS: Record<string, string> = {
  huggingface: 'openai-fast',
  gitee: 'DeepSeek-V3.2',
  modelscope: 'deepseek-ai/DeepSeek-V3.2'
};

const OPTIM_MODEL_STORAGE_PREFIX = 'optim_model_';

export const getOptimizationModel = (provider: string): string => {
  if (typeof localStorage === 'undefined') return DEFAULT_OPTIMIZATION_MODELS[provider] || 'openai-fast';
  return localStorage.getItem(OPTIM_MODEL_STORAGE_PREFIX + provider) || DEFAULT_OPTIMIZATION_MODELS[provider] || 'openai-fast';
};

export const saveOptimizationModel = (provider: string, model: string) => {
  if (typeof localStorage !== 'undefined') {
      const defaultModel = DEFAULT_OPTIMIZATION_MODELS[provider];
      // If saving default content or empty, remove the key to fallback to default
      if (model === defaultModel || !model.trim()) {
          localStorage.removeItem(OPTIM_MODEL_STORAGE_PREFIX + provider);
      } else {
          localStorage.setItem(OPTIM_MODEL_STORAGE_PREFIX + provider, model.trim());
      }
  }
};

// --- Video Settings Management ---

export interface VideoSettings {
  prompt: string;
  duration: number; // in seconds
  steps: number;
  guidance: number;
}

export const DEFAULT_VIDEO_SETTINGS: Record<string, VideoSettings> = {
  huggingface: {
    prompt: "make this image come alive, cinematic motion, smooth animation",
    duration: 3,
    steps: 6,
    guidance: 1
  },
  gitee: {
    prompt: "make this image come alive, cinematic motion, smooth animation",
    duration: 3,
    steps: 10,
    guidance: 4
  },
  modelscope: {
    prompt: "make this image come alive, cinematic motion, smooth animation",
    duration: 3,
    steps: 10,
    guidance: 4
  }
};

const VIDEO_SETTINGS_STORAGE_PREFIX = 'video_settings_';

export const getVideoSettings = (provider: string): VideoSettings => {
  const defaults = DEFAULT_VIDEO_SETTINGS[provider] || DEFAULT_VIDEO_SETTINGS['huggingface'];
  if (typeof localStorage === 'undefined') return defaults;
  
  try {
    const raw = localStorage.getItem(VIDEO_SETTINGS_STORAGE_PREFIX + provider);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw);
    // Ensure all keys exist by merging with defaults
    return { ...defaults, ...parsed };
  } catch {
    return defaults;
  }
};

export const saveVideoSettings = (provider: string, settings: VideoSettings) => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(VIDEO_SETTINGS_STORAGE_PREFIX + provider, JSON.stringify(settings));
  }
};

// --- Translation Service ---

const POLLINATIONS_API_URL = "https://text.pollinations.ai/openai";

export const translatePrompt = async (text: string): Promise<string> => {
    try {
        const response = await fetch(POLLINATIONS_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: 'openai-fast',
                messages: [
                    {
                        role: 'system',
                        content: `You are a professional language translation engine.
Your sole responsibility is to translate user-provided text into English. Before processing any input, you must first identify its original language.
If the input text is already in English, return the original English text directly without any modification. If the input text is not in English, translate it precisely into English.
Your output must strictly adhere to the following requirements: it must contain only the final English translation or the original English text, without any explanations, comments, descriptions, prefixes, suffixes, quotation marks, or other non-translated content.`
                    },
                    {
                        role: 'user',
                        content: text
                    }
                ],
                stream: false
            }),
        });

        if (!response.ok) {
            throw new Error("Translation request failed");
        }

        const data = await response.json();
        const content = data.choices?.[0]?.message?.content;

        return content || text;
    } catch (error) {
        console.error("Translation Error:", error);
        throw new Error("error_translation_failed");
    }
};

export const optimizeEditPrompt = async (imageBase64: string, prompt: string): Promise<string> => {
  try {
    // Pollinations AI OpenAI-compatible endpoint
    const response = await fetch(POLLINATIONS_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'openai-fast',
        messages: [
          {
            role: 'system',
            content: `You are a professional AI image editing assistant.
Your task is to analyze the image provided by the user (which may include user-drawn masks/indicated editing areas) and the user's text request, and deeply understand their intent.
When analyzing the image, you must actively extract and integrate its inherent visual context, including but not limited to the image subject, existing elements, color scheme, lighting conditions, and overall atmosphere, ensuring seamless integration with the optimized editing instructions.
Based on the visual context and text, optimize the user's editing instructions into more precise, descriptive prompts that are easier for the AI ​​model to understand.
When the user's request is vague or incomplete, intelligently infer and supplement specific, reasonable visual details to refine the editing instructions.
When generating optimized prompts, be sure to clearly incorporate descriptions of the expected visual changes, prioritizing the addition of detailed visual styles, precise lighting conditions, reasonable compositional layouts, and specific material textures to ensure the AI ​​model can accurately understand and execute the instructions.
For example: 'Replace the masked area with [specific object], emphasizing its [material], [color], and [lighting effect]', 'Add a [new object] at [specified location], giving it a [specific style] and [compositional relationship]', or 'Adjust the overall image style to [artistic style], keeping [original elements] unchanged, but enhancing [a certain feature]'.
Keep the generated prompts concise and descriptive, prioritizing the use of descriptive keywords and phrases that are easier for AI image models to understand and respond to, to maximize the effectiveness and accuracy of the prompt execution.
Only reply with the optimized prompt text. Do not add any conversational content. Do not include any markdown syntax. Ensure the output language matches the language of the prompt that needs to be optimized.`
          },
          {
            role: 'user',
            content: [
                { type: "text", text: prompt },
                { type: "image_url", image_url: { url: imageBase64 } }
            ]
          }
        ],
        stream: false
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to optimize prompt");
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    return content || prompt;
  } catch (error) {
    console.error("Optimize Edit Prompt Error:", error);
    throw error;
  }
};

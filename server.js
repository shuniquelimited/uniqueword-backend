import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import { GoogleGenAI, ThinkingLevel } from "@google/genai";

const app = express();
const PORT = Number(process.env.PORT) || 3000;

// ডেস্কটপ (.exe / Electron / localhost) এবং লাইভ ওয়েব সব অরিজিন থেকে রিকোয়েস্ট গ্রহণের জন্য CORS
app.use(
  cors({
    origin: true,
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
app.options("*", cors());

app.use(express.json({ limit: "20mb" }));

// multipart/form-data (অডিও ফাইল আপলোড) সাপোর্টের জন্য
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 }, // সর্বোচ্চ 15MB অডিও
});

function getGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

async function withRetry(fn, maxAttempts = 2) {
  let lastError;
  for (let i = 0; i < maxAttempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      if (!/\b(429|503|UNAVAILABLE|RESOURCE_EXHAUSTED|timed?\s*out)\b/i.test(msg) || i === maxAttempts - 1) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 500 * (i + 1)));
    }
  }
  throw lastError;
}

// ১) Health Check Route
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    status: "healthy",
    service: "Unique Word Cloud AI & Voice Backend",
    aiReady: Boolean(process.env.GEMINI_API_KEY?.trim()),
    uptimeSeconds: Math.round(process.uptime()),
  });
});

// ২) Text Generation, Editing, Summary & Translation Route (/api/ai)
// Body: { task?: "write"|"edit"|"summarize"|"translate"|"grammar"|"chat", prompt: string, content?: string, targetLanguage?: string, tone?: string }
app.post("/api/ai", async (req, res) => {
  try {
    const ai = getGeminiClient();
    if (!ai) {
      return res.status(503).json({
        error: "Server environment variable GEMINI_API_KEY is not configured.",
      });
    }

    const {
      task = "write",
      prompt = "",
      content = "",
      targetLanguage = "Bengali",
      tone = "Professional",
    } = req.body || {};

    if (!String(prompt).trim() && !String(content).trim()) {
      return res.status(400).json({ error: "Please provide 'prompt' or 'content'." });
    }

    const systemInstruction = `You are UNIQUE AI, the multilingual writing assistant for Unique Word.
You natively write and understand Bengali (বাংলা Unicode), Banglish (phonetic Bengali written in Latin letters), English, Arabic (العربية), and Urdu (اردو).
If the user writes in Banglish or asks for Bengali ("banglay", "বাংলায়"), always respond in natural Bengali Unicode.
Return clean, directly usable text or HTML (<p>, <h1>, <h2>, <ul>, <li>, <table>, <strong>) without markdown code fences.`;

    let userPrompt = "";
    switch (String(task).toLowerCase()) {
      case "edit":
      case "rewrite":
        userPrompt = `Rewrite and improve this text in a ${tone} tone. Instructions: ${prompt || "Improve clarity and flow"}\n\nText:\n${content}`;
        break;
      case "grammar":
        userPrompt = `Fix spelling, grammar, and punctuation while keeping the original meaning and language:\n\n${content || prompt}`;
        break;
      case "summarize":
      case "summary":
        userPrompt = `Provide a clear, structured summary with key points:\n\n${content || prompt}`;
        break;
      case "translate":
        userPrompt = `Translate accurately into ${targetLanguage} while preserving formatting and nuance:\n\n${content || prompt}`;
        break;
      default:
        userPrompt = content
          ? `Task/Prompt: ${prompt}\n\nDocument Context:\n${String(content).slice(0, 6000)}`
          : String(prompt);
    }

    const models = ["gemini-4-hydrogen", "gemini-3.1-flash-lite"];
    let result = null;
    let usedModel = models[0];

    for (let i = 0; i < models.length; i++) {
      try {
        usedModel = models[i];
        result = await withRetry(() =>
          ai.models.generateContent({
            model: usedModel,
            contents: userPrompt,
            config: {
              systemInstruction,
              ...(usedModel === "gemini-4-hydrogen"
                ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
                : {}),
            },
          })
        );
        break;
      } catch (err) {
        if (i === models.length - 1) throw err;
      }
    }

    res.json({
      ok: true,
      result: (result?.text ?? "").trim(),
      model: usedModel,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "AI request failed";
    res.status(500).json({ error: message });
  }
});

// ৩) Multilingual Voice-to-Text Route (/api/voice-to-text)
// Accepts JSON { audioBase64, mimeType, language } OR multipart/form-data field "audio"
app.post("/api/voice-to-text", upload.single("audio"), async (req, res) => {
  try {
    const ai = getGeminiClient();
    if (!ai) {
      return res.status(503).json({
        error: "Server environment variable GEMINI_API_KEY is not configured.",
      });
    }

    const language = String(req.body?.language || "bn-BD").trim();
    let audioBase64 = "";
    let mimeType = "audio/webm";

    if (req.file?.buffer) {
      audioBase64 = req.file.buffer.toString("base64");
      mimeType = req.file.mimetype || "audio/webm";
    } else if (req.body?.audioBase64) {
      const raw = String(req.body.audioBase64);
      const commaIdx = raw.indexOf(",");
      audioBase64 = commaIdx >= 0 ? raw.slice(commaIdx + 1) : raw;
      mimeType = String(req.body?.mimeType || "audio/webm").split(";")[0];
    }

    if (!audioBase64) {
      return res.status(400).json({ error: "No audio data received ('audio' file or 'audioBase64')." });
    }

    const langHintMap = {
      "bn-BD": "Bengali (বাংলা Unicode script)",
      "en-US": "English (US)",
      "ar-SA": "Arabic (العربية script)",
      "ur-PK": "Urdu (اردو script)",
      auto: "the spoken language (Bengali, English, Arabic, or Urdu)",
    };
    const targetLangDesc = langHintMap[language] || language;

    const response = await withRetry(() =>
      ai.models.generateContent({
        model: "gemini-3.5-transcribe",
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: audioBase64,
              },
            },
            {
              text: `Transcribe this spoken audio accurately in ${targetLangDesc}. Apply natural punctuation and return ONLY the transcribed text without extra commentary.`,
            },
          ],
        },
      })
    );

    res.json({
      ok: true,
      text: (response.text ?? "").trim(),
      language,
      model: "gemini-3.5-transcribe",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Voice transcription failed";
    res.status(500).json({ error: message });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Unique Word Backend running on port ${PORT}`);
});
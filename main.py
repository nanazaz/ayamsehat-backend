"""
ayamsehat.online — FastAPI Backend
- CNN  : Deteksi penyakit dari foto ayam ATAU foto kotoran ayam
- Groq : Validasi foto + estimasi usia + analisis visual
"""

import io, json, os, time, base64
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from groq import Groq, RateLimitError
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

load_dotenv()

DISEASE_MODEL_PATH = Path("./saved_model/best_model.h5")
DISEASE_LABEL_PATH = Path("./saved_model/label_map.json")
IMG_SIZE           = (224, 224)
GROQ_KEY           = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = "qwen/qwen3.6-27b"

DISEASE_INFO = {
    "Healthy": {
        "nama_id"    : "Sehat",
        "keparahan"  : "Tidak ada",
        "deskripsi"  : "Unggas dalam kondisi sehat.",
        "rekomendasi": [
            "Pertahankan pola pemberian pakan yang seimbang.",
            "Lakukan vaksinasi rutin sesuai jadwal.",
            "Jaga kebersihan kandang setiap hari.",
        ],
    },
    "Coccidiosis": {
        "nama_id"    : "Koksidiosis",
        "keparahan"  : "Sedang",
        "deskripsi"  : "Penyakit parasit protozoa yang menyerang usus unggas.",
        "rekomendasi": [
            "Berikan obat anticoccidial (Amprolium / Toltrazuril) sesuai dosis.",
            "Pisahkan unggas yang terinfeksi dari kawanan.",
            "Jaga lantai kandang tetap kering.",
            "Konsultasi dokter hewan jika tidak membaik dalam 3 hari.",
        ],
    },
    "Newcastle Disease": {
        "nama_id"    : "Penyakit Newcastle (ND/Tetelo)",
        "keparahan"  : "Berat",
        "deskripsi"  : "Penyakit viral sangat menular.",
        "rekomendasi": [
            "SEGERA isolasi semua unggas yang sakit.",
            "Vaksinasi ND untuk unggas yang belum terinfeksi.",
            "Disinfeksi kandang total.",
            "Hubungi dokter hewan dalam 24 jam.",
        ],
    },
    "Salmonella": {
        "nama_id"    : "Salmonellosis",
        "keparahan"  : "Sedang-Berat",
        "deskripsi"  : "Infeksi bakteri Salmonella. Bersifat zoonosis.",
        "rekomendasi": [
            "Gunakan APD saat menangani unggas sakit.",
            "Berikan antibiotik atas resep dokter hewan.",
            "Pisahkan unggas sakit dan sterilkan peralatan.",
            "Jangan konsumsi telur dari unggas yang sakit.",
        ],
    },
}

app = FastAPI(title="ayamsehat.online API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cnn_model = None

@app.on_event("startup")
async def load_model():
    global cnn_model
    if TF_AVAILABLE and DISEASE_MODEL_PATH.exists():
        print("🔄 Memuat CNN model...")
        cnn_model = tf.keras.models.load_model(str(DISEASE_MODEL_PATH))
        print("✅ CNN model berhasil dimuat!")
    else:
        print("⚠️  CNN model tidak ditemukan")
    if GROQ_KEY:
        print("✅ Groq API terkonfigurasi!")
    else:
        print("⚠️  GROQ_API_KEY tidak ditemukan")

def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

def load_label_map() -> dict:
    if DISEASE_LABEL_PATH.exists():
        with open(DISEASE_LABEL_PATH) as f:
            return json.load(f)
    return {}

def strip_reasoning(text: str) -> str:
    """Qwen (reasoning model) kadang mengeluarkan blok <think>...</think>
    sebelum jawaban akhir. Buang blok itu, ambil JSON setelahnya."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip().replace("```json", "").replace("```", "").strip()
    # Ambil bagian dari '{' pertama sampai '}' terakhir, jaga-jaga ada teks nyasar
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return text

def groq_call_with_retry(client, max_retries=3, **kwargs):
    """Panggil Groq API, retry otomatis kalau kena rate limit (429)."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 10 * (attempt + 1)
            print(f"⏳ Rate limit kena, tunggu {wait_time}s sebelum coba lagi... (percobaan {attempt+1}/{max_retries})")
            time.sleep(wait_time)

async def run_disease_prediction(img_bytes: bytes) -> dict:
    if cnn_model is None:
        raise HTTPException(503, "CNN model belum dimuat.")
    label_map = load_label_map()
    preds     = cnn_model.predict(preprocess(img_bytes), verbose=0)[0]
    top_idx   = int(np.argmax(preds))
    top_label = label_map.get(str(top_idx), "Unknown")
    top_conf  = float(preds[top_idx])
    all_classes = sorted([
        {"label": label_map.get(str(i), f"Class {i}"), "confidence": round(float(p)*100, 2)}
        for i, p in enumerate(preds)
    ], key=lambda x: x["confidence"], reverse=True)
    info = DISEASE_INFO.get(top_label, {})
    return {
        "label"      : top_label,
        "label_id"   : info.get("nama_id", top_label),
        "confidence" : round(top_conf * 100, 2),
        "keparahan"  : info.get("keparahan", "-"),
        "deskripsi"  : info.get("deskripsi", ""),
        "rekomendasi": info.get("rekomendasi", []),
        "all_classes": all_classes,
    }

async def run_groq_analysis(img_bytes, jenis_unggas, gejala, berat) -> dict:
    if not GROQ_KEY:
        return {
            "is_poultry"         : False,
            "pesan_validasi"     : "GROQ_API_KEY belum diisi di .env",
            "estimasi_usia"      : "-",
            "fase_pertumbuhan"   : "-",
            "rentang_normal_berat": "-",
            "kondisi_visual"     : [],
            "rekomendasi_khusus" : [],
            "catatan_khusus"     : ""
        }
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client  = Groq(api_key=GROQ_KEY)

        # STEP 1 — Validasi foto: unggas ATAU kotoran unggas
        val_resp = groq_call_with_retry(
            client,
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": 'Apakah gambar ini menunjukkan: (1) unggas/ayam/bebek/burung ternak, ATAU (2) kotoran/feses unggas ternak? Kedua jenis foto ini valid untuk diagnosa penyakit unggas. Jawab HANYA JSON: {"is_poultry": true/false, "tipe_foto": "unggas/kotoran/bukan_keduanya", "alasan": "penjelasan singkat bahasa Indonesia"}'},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}],
            max_tokens=1024,
        )
        val_raw = strip_reasoning(val_resp.choices[0].message.content)
        print("🔍 RAW GROQ VALIDATION RESPONSE:", repr(val_raw))
        try:
            val        = json.loads(val_raw)
            is_poultry = val.get("is_poultry", True)
            tipe_foto  = val.get("tipe_foto", "unggas")
            alasan     = val.get("alasan", "")
        except Exception as parse_err:
            print(f"⚠️ Gagal parse validasi JSON: {parse_err}")
            is_poultry = True
            tipe_foto  = "unggas"
            alasan     = ""

        # Kalau bukan unggas dan bukan kotoran unggas
        if not is_poultry:
            return {
                "is_poultry"         : False,
                "pesan_validasi"     : f"Foto ini bukan unggas atau kotoran unggas. {alasan} Mohon upload foto ayam/bebek/unggas atau foto kotorannya ya!",
                "estimasi_usia"      : "-",
                "fase_pertumbuhan"   : "-",
                "rentang_normal_berat": "-",
                "kondisi_visual"     : [],
                "rekomendasi_khusus" : [],
                "catatan_khusus"     : ""
            }

        # STEP 2 — Analisis sesuai tipe foto
        if tipe_foto == "kotoran":
            konteks_foto = "foto kotoran/feses unggas"
            instruksi_visual = """Analisis kondisi kotoran unggas:
- Warna kotoran (normal/abnormal)
- Konsistensi kotoran (padat/cair/berdarah)
- Ada tidaknya lendir atau darah
- Bau yang mungkin terindikasi dari tampilan
- Tanda-tanda infeksi yang terlihat"""
        else:
            konteks_foto = "foto unggas secara langsung"
            instruksi_visual = """Analisis kondisi fisik unggas:
- Kondisi bulu (bersih/kusam/rontok)
- Postur tubuh (tegak/membungkuk/lesu)
- Kondisi jengger/pial jika terlihat
- Kondisi mata dan paruh
- Tanda-tanda fisik abnormal lainnya"""

        prompt = f"""Kamu adalah dokter hewan ahli unggas ternak Indonesia.
Input adalah {konteks_foto}.
Jenis unggas : {jenis_unggas}
Berat        : {berat if berat else 'tidak diketahui'}
Gejala       : {gejala if gejala else 'tidak ada'}

{instruksi_visual}

PENTING: Jangan berpikir terlalu panjang. Langsung analisis singkat lalu keluarkan JSON.

Balas HANYA JSON ini (tanpa teks lain):
{{
  "is_poultry": true,
  "pesan_validasi": "",
  "tipe_foto": "{tipe_foto}",
  "estimasi_usia": "X minggu/bulan (jika bisa diestimasi dari foto, atau tulis 'Tidak dapat diestimasi dari kotoran')",
  "fase_pertumbuhan": "Starter/Grower/Finisher/Layer (atau '-' jika foto kotoran)",
  "rentang_normal_berat": "X-Y kg (atau '-' jika foto kotoran)",
  "kondisi_visual": ["temuan 1","temuan 2","temuan 3","temuan 4","temuan 5"],
  "rekomendasi_khusus": ["rekomendasi 1","rekomendasi 2","rekomendasi 3"],
  "catatan_khusus": "catatan tambahan"
}}"""

        resp = groq_call_with_retry(
            client,
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}],
            max_tokens=6000,
        )
        raw = strip_reasoning(resp.choices[0].message.content)
        print("🔍 RAW GROQ ANALYSIS RESPONSE:", repr(raw))
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            print("⚠️ JSON analisis kepotong/rusak, pakai fallback minimal")
            result = {
                "estimasi_usia"      : "-",
                "fase_pertumbuhan"   : "-",
                "rentang_normal_berat": "-",
                "kondisi_visual"     : [],
                "rekomendasi_khusus" : [],
                "catatan_khusus"     : "Analisis visual Groq tidak lengkap, coba upload ulang foto."
            }
        result.setdefault("is_poultry", True)
        result.setdefault("pesan_validasi", "")
        result.setdefault("tipe_foto", tipe_foto)
        return result

    except Exception as e:
        print(f"❌ Error Groq: {e}")
        return {
            "is_poultry"         : True,
            "pesan_validasi"     : "",
            "tipe_foto"          : "unggas",
            "estimasi_usia"      : "Gagal",
            "fase_pertumbuhan"   : "-",
            "rentang_normal_berat": "-",
            "kondisi_visual"     : [str(e)],
            "rekomendasi_khusus" : [],
            "catatan_khusus"     : ""
        }

class FullResponse(BaseModel):
    success        : bool
    processing_time: float
    diagnosa       : dict
    estimasi_usia  : Optional[dict] = None

@app.get("/health")
async def health():
    return {
        "status"    : "ok",
        "cnn_loaded": cnn_model is not None,
        "groq_ready": bool(GROQ_KEY),
    }

@app.post("/full-analysis", response_model=FullResponse)
async def full_analysis(
    foto        : UploadFile = File(...),
    jenis_unggas: str = Form("Ayam"),
    gejala      : str = Form(""),
    berat       : str = Form(""),
):
    t0        = time.time()
    img_bytes = await foto.read()
    disease   = await run_disease_prediction(img_bytes)
    groq      = await run_groq_analysis(img_bytes, jenis_unggas, gejala, berat)
    return FullResponse(
        success=True,
        processing_time=round(time.time()-t0, 3),
        diagnosa=disease,
        estimasi_usia=groq,
    )

@app.post("/predict")
async def predict_only(foto: UploadFile = File(...)):
    t0        = time.time()
    img_bytes = await foto.read()
    result    = await run_disease_prediction(img_bytes)
    return {"success": True, "processing_time": round(time.time()-t0,3), "diagnosa": result}
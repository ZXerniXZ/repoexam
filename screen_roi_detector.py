#!/usr/bin/env python3
"""
Screen ROI Detector
===================
Connects to IP Webcam, captures a frame on Enter press or hardware button,
detects the PC screen and saves the image with the ROI drawn.
Optional: GPIO button (27) to trigger capture; motor (17) for haptic feedback
(1 buzz = photo taken, N buzzes = answer 1–4).
"""


import cv2
import numpy as np
import sys
import time
import threading
import requests
import base64
import json
import os
import re
import select
from dotenv import load_dotenv

# Optional GPIO (Raspberry Pi): motor + button for capture and haptic feedback
try:
    from gpiozero import OutputDevice, Button
    _GPIO_AVAILABLE = True
except Exception:
    OutputDevice = None
    Button = None
    _GPIO_AVAILABLE = False

# Load environment variables
load_dotenv()

# ── IP Webcam configuration ─────────────────────────────────────────
IP_WEBCAM_IP   = "10.22.179.8"
IP_WEBCAM_PORT = 8080
SHOT_URL = f"http://{IP_WEBCAM_IP}:{IP_WEBCAM_PORT}/shot.jpg"

# ── GPIO (vibrazione + pulsante) ────────────────────────────────────
MOTOR_GPIO = 17
BUTTON_GPIO = 27
VIBE_CONFIRM_S = 0.15
VIBE_ON_S = 0.15
VIBE_OFF_S = 0.25
# Workflow: si avvia al RILASCIO del pulsante; si può ricatturare solo dopo che
# l'analisi OpenRouter è terminata E l'utente ha premuto di nuovo (arm) e rilasciato.
# (Nessun debounce extra: la sequenza premuto→rilasciato→run→premuto→rilasciato è già robusta.)

# ── OpenRouter configuration ─────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")
OPENROUTER_PROMPT = (
    "analizza questa domanda e rispondi da 1 a 4 con la risposta corretta, "
    "dove la 1 è quella più alta la 2 la seconda e cosi via, "
    "rispondi solo con il numero della domanda"
)

# ── Detection parameters ────────────────────────────────────────────
CANNY_LOW          = 30
CANNY_HIGH         = 100
MIN_AREA_RATIO     = 0.04
MAX_AREA_RATIO     = 0.92
POLY_EPSILON       = 0.02
ASPECT_RATIO_RANGE = (1.0, 2.8)
MORPH_KERNEL_SIZE  = 5
MIN_BRIGHTNESS      = 0.15  # Screens should be relatively bright

# ── Text enhancement parameters ───────────────────────────────────────
# Parametri configurabili per migliorare la leggibilità del testo
ENHANCE_ENABLED     = True   # Abilita/disabilita enhancement
ENHANCE_BILATERAL_D = 7      # Bilateral filter: diameter (1-15, dispari)
ENHANCE_BILATERAL_SIGMA_COLOR = 50   # Bilateral: sigma color (10-100)
ENHANCE_BILATERAL_SIGMA_SPACE = 50   # Bilateral: sigma space (10-100)
ENHANCE_CLAHE_CLIP  = 4.0    # CLAHE clip limit (1.0-8.0, più alto = più contrasto)
ENHANCE_CLAHE_TILE  = 8      # CLAHE tile grid size (4-16, più grande = più uniforme)
ENHANCE_PERCENTILE_LOW = 1   # Percentile inferiore per normalizzazione (0-5)
ENHANCE_PERCENTILE_HIGH = 99 # Percentile superiore per normalizzazione (95-100)
ENHANCE_UNSHARP_AMOUNT = 1.5 # Unsharp mask: amount (0.5-3.0, più alto = più nitido)
ENHANCE_UNSHARP_SIGMA = 2.0  # Unsharp mask: sigma blur (0.5-5.0)
ENHANCE_GAMMA = 1.0          # Gamma correction (0.5-2.0, <1 = più chiaro, >1 = più scuro)


class ScreenROIDetector:
    """Detect a PC screen in a captured frame and draw a ROI around it."""

    def __init__(self, shot_url: str):
        self.shot_url = shot_url
        self.roi: np.ndarray | None = None
        self.canny_low  = CANNY_LOW
        self.canny_high = CANNY_HIGH
        self._motor = None
        self._button = None
        
        # Text enhancement parameters
        self.enhance_enabled = ENHANCE_ENABLED
        self.enhance_bilateral_d = ENHANCE_BILATERAL_D
        self.enhance_bilateral_sigma_color = ENHANCE_BILATERAL_SIGMA_COLOR
        self.enhance_bilateral_sigma_space = ENHANCE_BILATERAL_SIGMA_SPACE
        self.enhance_clahe_clip = ENHANCE_CLAHE_CLIP
        self.enhance_clahe_tile = ENHANCE_CLAHE_TILE
        self.enhance_percentile_low = ENHANCE_PERCENTILE_LOW
        self.enhance_percentile_high = ENHANCE_PERCENTILE_HIGH
        self.enhance_unsharp_amount = ENHANCE_UNSHARP_AMOUNT
        self.enhance_unsharp_sigma = ENHANCE_UNSHARP_SIGMA
        self.enhance_gamma = ENHANCE_GAMMA

    # ── Capture ─────────────────────────────────────────────────────
    MAX_RETRIES = 3

    def capture(self) -> np.ndarray | None:
        """
        Grab a single JPEG frame from IP Webcam.
        Uses requests with:
          - Connection: close (no keep-alive → avoids stale sockets)
          - Retry with exponential backoff on failure
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    self.shot_url,
                    timeout=15,
                    headers={"Connection": "close"},
                    stream=False,
                )
                resp.raise_for_status()
                img_array = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                print(f"[WARN] Tentativo {attempt}: frame vuoto, riprovo...")
            except requests.RequestException as e:
                print(f"[WARN] Tentativo {attempt}/{self.MAX_RETRIES}: {e}")
            # Backoff before retry
            if attempt < self.MAX_RETRIES:
                wait = attempt * 1.5
                print(f"       Attendo {wait:.1f}s prima di riprovare...")
                time.sleep(wait)

        print("[ERROR] Tutti i tentativi falliti.")
        return None

    # ── Preprocessing ───────────────────────────────────────────────
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """Preprocess for edge detection: denoise + enhance contrast."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        # Enhance local contrast (helps with uneven lighting)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    # ── Edge map ────────────────────────────────────────────────────
    def _edge_map(self, gray: np.ndarray) -> np.ndarray:
        """Canny edge detection + morphological closing to connect gaps."""
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        # Close small gaps in edges
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    # ── Candidate scoring ───────────────────────────────────────────
    @staticmethod
    def _score_candidate(
        contour: np.ndarray,
        gray: np.ndarray,
        frame_area: int,
        frame_center: tuple[float, float],
    ) -> float:
        """Score a contour: higher = more likely to be a screen."""
        area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            return 0.0

        aspect = max(w, h) / min(w, h)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0

        # Average brightness inside the contour (screens are bright)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        brightness = cv2.mean(gray, mask=mask)[0] / 255.0

        # Reject if too dark (not a screen)
        if brightness < MIN_BRIGHTNESS:
            return 0.0

        # Distance from frame center (prefer centered screens)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        max_dist = np.hypot(frame_center[0], frame_center[1])
        center_dist = np.hypot(cx - frame_center[0], cy - frame_center[1])
        centrality = 1.0 - (center_dist / max_dist)

        # Weighted score
        score = (
            0.35 * (area / frame_area)     # Prefer large
          + 0.25 * rectangularity          # Prefer rectangular
          + 0.25 * brightness              # Prefer bright (screens!)
          + 0.10 * centrality              # Prefer centered
          + 0.05 * (1.0 - abs(aspect - 1.78) / 1.78)  # Prefer ~16:9
        )
        return score

    # ── Find the real 4 corners from a contour ───────────────────────
    @staticmethod
    def _find_four_corners(contour: np.ndarray) -> np.ndarray | None:
        """
        Extract the 4 real perspective corners from a contour.
        Strategy 1: approxPolyDP with progressive epsilon
        Strategy 2: convex hull → pick 4 extreme corners
        """
        peri = cv2.arcLength(contour, True)

        # Strategy 1: progressively simplify until we get 4 vertices
        for eps_factor in np.arange(0.01, 0.10, 0.005):
            approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.int32)
                # Validate: all 4 corners must be distinct
                if ScreenROIDetector._validate_corners(corners):
                    return corners

        # Strategy 2: convex hull → pick the 4 most extreme corner points
        hull = cv2.convexHull(contour)
        hull_pts = hull.reshape(-1, 2).astype(np.float64)

        if len(hull_pts) < 4:
            return None

        # Find the 4 corners using sum/diff of coordinates
        s = hull_pts.sum(axis=1)      # x + y
        d = hull_pts[:, 0] - hull_pts[:, 1]  # x - y

        corners = np.zeros((4, 2), dtype=np.int32)
        corners[0] = hull_pts[np.argmin(s)].astype(np.int32)   # top-left
        corners[1] = hull_pts[np.argmin(d)].astype(np.int32)   # top-right
        corners[2] = hull_pts[np.argmax(s)].astype(np.int32)   # bottom-right
        corners[3] = hull_pts[np.argmax(d)].astype(np.int32)   # bottom-left

        # Validate corners
        if not ScreenROIDetector._validate_corners(corners):
            return None

        return corners

    @staticmethod
    def _validate_corners(corners: np.ndarray) -> bool:
        """Ensure all 4 corners are distinct and form a valid quadrilateral."""
        if corners.shape != (4, 2):
            return False

        # Check all corners are distinct (min distance > 10 pixels)
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(corners[i] - corners[j])
                if dist < 10:
                    return False  # Duplicate corner

        # Check corners form a reasonable quadrilateral (not all collinear)
        # Compute area of the quadrilateral
        area = cv2.contourArea(corners.reshape(-1, 1, 2))
        if area < 100:  # Too small
            return False

        return True

    # ── Main detection ──────────────────────────────────────────────
    def detect_screen(self, frame: np.ndarray) -> np.ndarray | None:
        """Detect the screen and return its 4 corner points."""
        gray = self._preprocess(frame)
        edges = self._edge_map(gray)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = frame.shape[0] * frame.shape[1]
        frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)

        best_score = 0.0
        best_corners = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frame_area * MIN_AREA_RATIO:
                continue
            if area > frame_area * MAX_AREA_RATIO:
                continue

            # Check rough shape: should be roughly quadrilateral
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, POLY_EPSILON * peri, True)
            if not (4 <= len(approx) <= 12):
                continue

            # Aspect ratio check via bounding rect
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if not (ASPECT_RATIO_RANGE[0] <= aspect <= ASPECT_RATIO_RANGE[1]):
                continue

            score = self._score_candidate(cnt, gray, frame_area, frame_center)
            if score > best_score:
                corners = self._find_four_corners(cnt)
                if corners is not None:
                    best_score = score
                    best_corners = corners

        return best_corners

    # ── Order points ────────────────────────────────────────────────
    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 perspective points: TL, TR, BR, BL.
        Ensures consistent ordering for perspective transform.
        """
        pts = pts.reshape(4, 2).astype(np.float64)
        rect = np.zeros((4, 2), dtype=np.int32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)].astype(np.int32)   # top-left:     smallest x+y
        rect[2] = pts[np.argmax(s)].astype(np.int32)   # bottom-right: largest  x+y
        
        d = pts[:, 0] - pts[:, 1]     # x - y
        rect[1] = pts[np.argmin(d)].astype(np.int32)   # top-right:    smallest x-y
        rect[3] = pts[np.argmax(d)].astype(np.int32)   # bottom-left:  largest  x-y
        
        return rect

    # ── Drawing ─────────────────────────────────────────────────────
    @staticmethod
    def _draw_roi(frame: np.ndarray, pts: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
        """Draw ROI overlay on frame."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)
        for i, pt in enumerate(pts):
            p = tuple(pt.ravel())
            cv2.circle(frame, p, 8, (0, 0, 255), -1)
            cv2.circle(frame, p, 8, (255, 255, 255), 2)
        return frame

    # ── Text enhancement ──────────────────────────────────────────────
    def enhance_text(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance text readability using a multi-stage pipeline:
        1. Bilateral filter (denoise while preserving edges)
        2. CLAHE (adaptive contrast enhancement)
        3. Percentile normalization (stretch dynamic range)
        4. Gamma correction (adjust brightness curve)
        5. Unsharp mask (sharpen text edges)
        
        Returns enhanced image (BGR format).
        """
        if not self.enhance_enabled:
            return image
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Bilateral filter: denoise while preserving text edges
        if self.enhance_bilateral_d > 0:
            # Ensure d is odd
            d = self.enhance_bilateral_d if self.enhance_bilateral_d % 2 == 1 else self.enhance_bilateral_d + 1
            gray = cv2.bilateralFilter(
                gray,
                d=d,
                sigmaColor=self.enhance_bilateral_sigma_color,
                sigmaSpace=self.enhance_bilateral_sigma_space
            )
        
        # 2. CLAHE: adaptive contrast enhancement
        if self.enhance_clahe_clip > 0:
            clahe = cv2.createCLAHE(
                clipLimit=self.enhance_clahe_clip,
                tileGridSize=(self.enhance_clahe_tile, self.enhance_clahe_tile)
            )
            gray = clahe.apply(gray)
        
        # 3. Percentile normalization: stretch dynamic range
        if self.enhance_percentile_low < self.enhance_percentile_high:
            low_val = np.percentile(gray, self.enhance_percentile_low)
            high_val = np.percentile(gray, self.enhance_percentile_high)
            if high_val > low_val:
                gray = np.clip(
                    (gray - low_val) * 255.0 / (high_val - low_val),
                    0, 255
                ).astype(np.uint8)
        
        # 4. Gamma correction
        if self.enhance_gamma != 1.0:
            inv_gamma = 1.0 / self.enhance_gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")
            gray = cv2.LUT(gray, table)
        
        # 5. Unsharp mask: sharpen text edges
        if self.enhance_unsharp_amount > 0 and self.enhance_unsharp_sigma > 0:
            blurred = cv2.GaussianBlur(gray, (0, 0), self.enhance_unsharp_sigma)
            gray = cv2.addWeighted(
                gray, 1.0 + self.enhance_unsharp_amount,
                blurred, -self.enhance_unsharp_amount,
                0
            )
        
        # Convert back to BGR (3 channels)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return result

    # ── OpenRouter analysis ──────────────────────────────────────────
    @staticmethod
    def analyze_with_openrouter(image_path: str) -> str | None:
        """
        Send image to OpenRouter API for analysis.
        Returns the response text or None on error.
        """
        if not OPENROUTER_API_KEY:
            print("[ERROR] OPENROUTER_API_KEY non configurata nel .env")
            return None

        # Read and encode image as base64
        try:
            # Read image with OpenCV to potentially resize if too large
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Impossibile leggere immagine: {image_path}")
                return None
            
            # Some models have size limits, resize if too large (max 2048px on longest side)
            h, w = img.shape[:2]
            max_dim = 2048
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"      Immagine ridimensionata: {w}x{h} → {new_w}x{new_h}")
            
            # Encode as JPEG
            success, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                print("[ERROR] Errore codifica JPEG")
                return None
            
            image_bytes = jpeg_buf.tobytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
        except FileNotFoundError:
            print(f"[ERROR] File non trovato: {image_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Errore lettura/elaborazione immagine: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ares/localRepo/repoexam",  # Richiesto da OpenRouter
            "X-Title": "Screen ROI Detector",  # Opzionale ma utile
        }

        payload = {
            "model": OPENROUTER_MODEL,
            "max_tokens": 500,  # Aumentato per evitare truncamento (il modello usa reasoning)
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OPENROUTER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "auto"  # Alcuni modelli richiedono questo
                            },
                        },
                    ],
                }
            ],
        }

        try:
            print(f"[...] Invio a OpenRouter (modello: {OPENROUTER_MODEL})...")
            print(f"      Dimensione immagine: {len(image_bytes) / 1024:.1f} KB")
            print(f"      Prompt: {OPENROUTER_PROMPT[:80]}...")
            
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,  # Aumentato timeout per modelli più lenti
            )
            
            # Debug: mostra status code
            print(f"      Status code: {resp.status_code}")
            
            resp.raise_for_status()
            data = resp.json()

            # Debug: mostra struttura completa della risposta
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                message = choice.get("message", {})
                
                # Controlla se c'è finish_reason
                finish_reason = choice.get("finish_reason", "unknown")
                print(f"      Finish reason: {finish_reason}")
                
                # Prova prima con content
                content = message.get("content", "")
                if content and content.strip():
                    result = content.strip()
                    print(f"[OK] Risposta ricevuta: {result}")
                    return result
                
                # Se content è vuoto, prova con reasoning (alcuni modelli usano questo)
                reasoning = message.get("reasoning", "")
                if reasoning and reasoning.strip():
                    reasoning_text = reasoning.strip()
                    
                    # Cerca pattern specifici per la risposta (es. "risposta: 1", "la risposta è 2", etc.)
                    patterns = [
                        r'(?:risposta|answer|soluzione|corretta|corretto)\s*[:\-]?\s*([1-4])',
                        r'(?:è|is)\s+([1-4])',
                        r'opzione\s+([1-4])',
                        r'numero\s+([1-4])',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, reasoning_text, re.IGNORECASE)
                        if match:
                            result = match.group(1)
                            print(f"[OK] Risposta estratta da reasoning (pattern: {pattern}): {result}")
                            return result
                    
                    # Fallback: cerca qualsiasi numero da 1 a 4
                    numbers = re.findall(r'\b([1-4])\b', reasoning_text)
                    if numbers:
                        result = numbers[-1]  # Prendi l'ultimo numero trovato
                        print(f"[OK] Risposta estratta da reasoning (ultimo numero): {result}")
                        print(f"      Reasoning snippet: {reasoning_text[-300:]}...")
                        return result
                    
                    # Se non trova numeri, mostra warning
                    print(f"[WARN] Nessun numero da 1 a 4 trovato nel reasoning")
                    print(f"      Ultima parte reasoning: {reasoning_text[-200:]}...")
                
                # Se entrambi sono vuoti, mostra warning
                if not content and not reasoning:
                    print(f"[WARN] Sia 'content' che 'reasoning' sono vuoti")
                elif finish_reason == "length":
                    print(f"[WARN] Risposta troncata (finish_reason=length)")
                    print(f"      Aumenta max_tokens o usa un modello più capace")
                    if reasoning:
                        print(f"      Reasoning parziale: {reasoning[:300]}...")
                
                # Mostra struttura per debug
                print(f"[DEBUG] Message structure: {json.dumps(message, indent=2)[:800]}...")
                return None

            print(f"[ERROR] Risposta inattesa: {json.dumps(data, indent=2)}")
            return None

        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP Error {e.response.status_code}: {e}")
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"    Dettagli errore:")
                    print(f"    {json.dumps(error_data, indent=4)}")
                    # Suggerimenti basati sull'errore
                    if "model" in str(error_data).lower() or "vision" in str(error_data).lower():
                        print()
                        print("    💡 Suggerimento: Il modello potrebbe non supportare le immagini.")
                        print("    Prova un modello vision come: openai/gpt-4o o google/gemini-pro-vision")
                except Exception:
                    print(f"    Body: {e.response.text[:500]}")
            return None
            
        except requests.exceptions.Timeout:
            print("[ERROR] Timeout: la richiesta ha impiegato troppo tempo (>60s)")
            print("    💡 Suggerimento: Prova un modello più veloce o riduci la dimensione dell'immagine")
            return None
            
        except requests.RequestException as e:
            print(f"[ERROR] Errore API OpenRouter: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"    Dettagli: {json.dumps(error_data, indent=2)}")
                except Exception:
                    print(f"    Body: {e.response.text[:500]}")
            return None

    # ── Vibrazione (conferma foto + risultato 1–4) ───────────────────
    def _vibrate_once(self):
        """Una breve vibrazione per confermare che la foto è stata scattata."""
        if self._motor is None:
            return
        try:
            self._motor.on()
            time.sleep(VIBE_CONFIRM_S)
            self._motor.off()
        except Exception:
            pass

    def _vibrate_n_times(self, n: int):
        """N vibrazioni brevi (risultato 1–4)."""
        if self._motor is None or n not in (1, 2, 3, 4):
            return
        try:
            for i in range(n):
                self._motor.on()
                time.sleep(VIBE_ON_S)
                self._motor.off()
                if i < n - 1:
                    time.sleep(VIBE_OFF_S)
        except Exception:
            pass

    @staticmethod
    def _parse_result_count(result: str | None) -> int | None:
        """Estrae un numero 1–4 dalla risposta OpenRouter."""
        if not result or not result.strip():
            return None
        s = result.strip()
        numbers = re.findall(r'\b([1-4])\b', s)
        if numbers:
            return int(numbers[-1])
        try:
            n = int(s)
            return n if 1 <= n <= 4 else None
        except ValueError:
            return None

    def _do_capture_and_analyze(self):
        """
        Cattura, rileva schermo, salva, vibra una volta, analizza con OpenRouter,
        poi vibra N volte in base alla risposta (1–4).
        """
        print(f"[...] Cattura da {self.shot_url} ...")
        frame = self.capture()
        if frame is None:
            print("[ERROR] Impossibile catturare. Controlla IP Webcam.")
            return

        print(f"[OK] Frame catturato: {frame.shape[1]}x{frame.shape[0]}")

        box = self.detect_screen(frame)
        if box is not None:
            self.roi = self._order_points(box)

            if not self._validate_corners(self.roi):
                print("[!] ROI non valida (corner duplicati).")
                cv2.imwrite("last_capture.jpg", frame)
                self._vibrate_once()
                return

            print("[OK] Schermo rilevato! Coordinate ROI:")
            labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            for label, pt in zip(labels, self.roi):
                print(f"     {label}: ({pt[0]}, {pt[1]})")

            output = self._draw_roi(frame.copy(), self.roi)
            cv2.imwrite("last_capture.jpg", output)
            print("[SAVE] Immagine con ROI → last_capture.jpg")

            warped = self._warp_perspective(frame, self.roi)
            if self.enhance_enabled:
                print("[ENHANCE] Applicazione miglioramento testo...")
                enhanced = self.enhance_text(warped)
                cv2.imwrite("screen_cropped.jpg", enhanced)
                print(f"[SAVE] Schermo raddrizzato + enhanced ({enhanced.shape[1]}x{enhanced.shape[0]}) → screen_cropped.jpg")
            else:
                cv2.imwrite("screen_cropped.jpg", warped)
                print(f"[SAVE] Schermo raddrizzato ({warped.shape[1]}x{warped.shape[0]}) → screen_cropped.jpg")

            self._vibrate_once()

            result = self.analyze_with_openrouter("screen_cropped.jpg")
            if result:
                print()
                print("  ╔══════════════════════════════════════════════╗")
                print(f"  ║  RISPOSTA:  {result:<34}║")
                print("  ╚══════════════════════════════════════════════╝")
                print()
                n = self._parse_result_count(result)
                if n is not None:
                    self._vibrate_n_times(n)
            else:
                print("[!] Analisi OpenRouter fallita.")
        else:
            self.roi = None
            cv2.imwrite("last_capture.jpg", frame)
            print("[!] Schermo non rilevato. Immagine salvata senza ROI → last_capture.jpg")
            print("    Prova +/- per regolare la sensibilità.")
            self._vibrate_once()

    # ── Perspective correction ──────────────────────────────────────
    @staticmethod
    def _warp_perspective(frame: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """
        Warp the perspective ROI into a flat rectangle.
        Output size is computed from the max width/height of the quadrilateral.
        """
        tl, tr, br, bl = roi.astype(np.float32)

        # Compute output width: max of top edge and bottom edge
        width_top    = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        out_w = int(max(width_top, width_bottom))

        # Compute output height: max of left edge and right edge
        height_left  = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        out_h = int(max(height_left, height_right))

        # Ensure landscape orientation (width > height for monitors)
        if out_h > out_w:
            # Swap if portrait, but keep the correct aspect
            out_w, out_h = out_h, out_w
            # Reorder points for swapped dimensions
            dst = np.array([
                [0,       0],
                [0,       out_h - 1],
                [out_w - 1, out_h - 1],
                [out_w - 1, 0],
            ], dtype=np.float32)
        else:
            dst = np.array([
                [0,       0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0,       out_h - 1],
            ], dtype=np.float32)

        # Source points (must match dst order: TL, TR, BR, BL)
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, M, (out_w, out_h))
        return warped

    # ── Parameter management ──────────────────────────────────────────
    def show_enhance_params(self):
        """Display current text enhancement parameters."""
        print()
        print("  ⚙️  Parametri Enhancement Testo:")
        print(f"    Abilitato              : {'✓' if self.enhance_enabled else '✗'}")
        print(f"    Bilateral d            : {self.enhance_bilateral_d} (1-15, dispari)")
        print(f"    Bilateral sigma color  : {self.enhance_bilateral_sigma_color} (10-100)")
        print(f"    Bilateral sigma space  : {self.enhance_bilateral_sigma_space} (10-100)")
        print(f"    CLAHE clip limit       : {self.enhance_clahe_clip:.1f} (1.0-8.0)")
        print(f"    CLAHE tile size        : {self.enhance_clahe_tile} (4-16)")
        print(f"    Percentile low         : {self.enhance_percentile_low} (0-5)")
        print(f"    Percentile high        : {self.enhance_percentile_high} (95-100)")
        print(f"    Unsharp amount         : {self.enhance_unsharp_amount:.1f} (0.5-3.0)")
        print(f"    Unsharp sigma          : {self.enhance_unsharp_sigma:.1f} (0.5-5.0)")
        print(f"    Gamma                  : {self.enhance_gamma:.2f} (0.5-2.0)")
        print()

    def modify_enhance_params(self):
        """Interactive parameter modification."""
        self.show_enhance_params()
        print("  Modifica parametri (INVIO per saltare):")
        print()

        # Enable/disable
        try:
            val = input(f"  Abilitato [{self.enhance_enabled}] (y/n): ").strip().lower()
            if val in ('y', 'yes', '1', 'true'):
                self.enhance_enabled = True
            elif val in ('n', 'no', '0', 'false'):
                self.enhance_enabled = False
        except (EOFError, KeyboardInterrupt):
            pass

        # Bilateral filter
        try:
            val = input(f"  Bilateral d [{self.enhance_bilateral_d}]: ").strip()
            if val:
                d = int(val)
                if 1 <= d <= 15:
                    self.enhance_bilateral_d = d if d % 2 == 1 else d + 1
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        try:
            val = input(f"  Bilateral sigma color [{self.enhance_bilateral_sigma_color}]: ").strip()
            if val:
                s = int(val)
                if 10 <= s <= 100:
                    self.enhance_bilateral_sigma_color = s
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        try:
            val = input(f"  Bilateral sigma space [{self.enhance_bilateral_sigma_space}]: ").strip()
            if val:
                s = int(val)
                if 10 <= s <= 100:
                    self.enhance_bilateral_sigma_space = s
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        # CLAHE
        try:
            val = input(f"  CLAHE clip limit [{self.enhance_clahe_clip:.1f}]: ").strip()
            if val:
                c = float(val)
                if 1.0 <= c <= 8.0:
                    self.enhance_clahe_clip = c
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        try:
            val = input(f"  CLAHE tile size [{self.enhance_clahe_tile}]: ").strip()
            if val:
                t = int(val)
                if 4 <= t <= 16:
                    self.enhance_clahe_tile = t
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        # Percentiles
        try:
            val = input(f"  Percentile low [{self.enhance_percentile_low}]: ").strip()
            if val:
                p = int(val)
                if 0 <= p <= 5:
                    self.enhance_percentile_low = p
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        try:
            val = input(f"  Percentile high [{self.enhance_percentile_high}]: ").strip()
            if val:
                p = int(val)
                if 95 <= p <= 100:
                    self.enhance_percentile_high = p
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        # Unsharp mask
        try:
            val = input(f"  Unsharp amount [{self.enhance_unsharp_amount:.1f}]: ").strip()
            if val:
                a = float(val)
                if 0.5 <= a <= 3.0:
                    self.enhance_unsharp_amount = a
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        try:
            val = input(f"  Unsharp sigma [{self.enhance_unsharp_sigma:.1f}]: ").strip()
            if val:
                s = float(val)
                if 0.5 <= s <= 5.0:
                    self.enhance_unsharp_sigma = s
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        # Gamma
        try:
            val = input(f"  Gamma [{self.enhance_gamma:.2f}]: ").strip()
            if val:
                g = float(val)
                if 0.5 <= g <= 2.0:
                    self.enhance_gamma = g
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        print()
        print("  ✓ Parametri aggiornati!")
        self.show_enhance_params()

    # ── Main loop ───────────────────────────────────────────────────
    def run(self):
        if _GPIO_AVAILABLE:
            try:
                self._motor = OutputDevice(MOTOR_GPIO)
                self._button = Button(BUTTON_GPIO, pull_up=True)
                print("[GPIO] Pulsante e motore inizializzati (pin {} e {})".format(BUTTON_GPIO, MOTOR_GPIO))
            except Exception as e:
                print("[GPIO] Non disponibile:", e)
                self._motor = None
                self._button = None
        else:
            self._motor = None
            self._button = None

        capture_lock = threading.Lock()
        # Stato visibile dal thread di capture (per riarmare solo a run finita)
        state_ref = ["WAIT_FOR_PRESS"]  # WAIT_FOR_PRESS | ARMED | RUNNING

        def run_capture_and_then_wait_for_press():
            try:
                self._do_capture_and_analyze()
            finally:
                capture_lock.release()
                state_ref[0] = "WAIT_FOR_PRESS"

        def start_capture_on_release():
            if not capture_lock.acquire(blocking=False):
                return
            print("[PULSANTE] Rilascio rilevato → avvio cattura e analisi.", flush=True)
            t = threading.Thread(target=run_capture_and_then_wait_for_press, daemon=True)
            t.start()

        if self._button is not None:
            self._button.when_pressed = lambda: None
            print("[GPIO] Stato pulsante all'avvio (is_pressed):", self._button.is_pressed, flush=True)
            print("[GPIO] Workflow: trigger al RILASCIO; riarmo dopo fine analisi + nuova PREMUTA.", flush=True)

        print()
        print("╔═════════════════════════════════════════════════════╗")
        print("║          Screen ROI Detector  v1.5                  ║")
        print("╠═════════════════════════════════════════════════════╣")
        print("║  PULSANTE o ENTER → Cattura + vibra + analizza AI   ║")
        print("║  +/-    → Regola sensibilità Canny                  ║")
        print("║  e      → Mostra parametri enhancement              ║")
        print("║  m      → Modifica parametri enhancement            ║")
        print("║  q      → Esci                                     ║")
        print("╚═════════════════════════════════════════════════════╝")
        print()
        print("  last_capture.jpg     → immagine originale con ROI")
        print("  screen_cropped.jpg   → schermo raddrizzato + enhanced")
        print()
        if OPENROUTER_API_KEY:
            print(f"  ✓ OpenRouter configurato (modello: {OPENROUTER_MODEL})")
        else:
            print("  ⚠ OpenRouter non configurato (manca OPENROUTER_API_KEY nel .env)")
        print()

        last_pressed = False if self._button is None else self._button.is_pressed
        need_prompt = True
        while True:
            if need_prompt:
                print("[>>] Premi ENTER per catturare (q=esci): ", end="", flush=True)
                need_prompt = False
            if select.select([sys.stdin], [], [], 0.2)[0]:
                try:
                    cmd = sys.stdin.readline().strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
            else:
                cmd = None
                if self._button is not None:
                    cur = self._button.is_pressed
                    s = state_ref[0]
                    if s == "WAIT_FOR_PRESS":
                        if not cur and last_pressed:
                            state_ref[0] = "ARMED"
                            print("[PULSANTE] Premuto (segnale perso) → armato.", flush=True)
                    elif s == "ARMED":
                        if cur and not last_pressed:
                            state_ref[0] = "RUNNING"
                            start_capture_on_release()
                    # RUNNING: il thread imposterà WAIT_FOR_PRESS quando ha finito
                    last_pressed = cur
                if cmd is None:
                    continue

            if cmd == "q":
                break

            elif cmd == "e":
                self.show_enhance_params()
                need_prompt = True
                continue

            elif cmd == "m":
                self.modify_enhance_params()
                need_prompt = True
                continue

            elif cmd in ("+", "="):
                self.canny_low  = min(self.canny_low + 5, 200)
                self.canny_high = min(self.canny_high + 10, 400)
                print(f"[CANNY] {self.canny_low} / {self.canny_high}")
                need_prompt = True
                continue

            elif cmd in ("-", "_"):
                self.canny_low  = max(self.canny_low - 5, 5)
                self.canny_high = max(self.canny_high - 10, 20)
                print(f"[CANNY] {self.canny_low} / {self.canny_high}")
                need_prompt = True
                continue

            # ── ENTER → capture + detect + warp + save + vibra + analizza ─
            capture_lock.acquire()
            try:
                self._do_capture_and_analyze()
            finally:
                capture_lock.release()
            need_prompt = True

        if self._motor is not None:
            try:
                self._motor.close()
            except Exception:
                pass
        print("[INFO] Chiuso.")


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = ScreenROIDetector(SHOT_URL)
    detector.run()


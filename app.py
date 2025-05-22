import streamlit as st
import pandas as pd
import os
import tempfile
import shutil
from datetime import datetime, date, timedelta
import re

# --- Dependencias de Gemini ---
import google.generativeai as genai
from PIL import Image  # Para el análisis de imágenes
import time  # Para los time.sleep

# --- MODELOS GEMINI  ---
TEXT_MODEL_NAME_GEMINI = 'gemini-2.0-flash'
VISION_MODEL_NAME_GEMINI = 'gemini-2.0-flash'


# Funcion para Parseo del Chat
def parse_whatsapp_chat_final(txt_filepath, media_folder_path):
    message_pattern = re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})\u202f(AM|PM)"
        r"\s+-\s+"
        r"([^:]+): "
        r"(.*)", re.IGNORECASE | re.UNICODE
    )
    media_explicit_pattern = re.compile(
        r"^([\w\d\s\-_\.]+\.(?:jpg|jpeg|png|gif|webp|mp4|avi|mov|opus|ogg|mp3|aac|pdf|docx?|xlsx?|pptx?))"
        r"\s+\(file attached\)$", re.IGNORECASE
    )
    media_omitted_marker = "<Media omitted>"
    edited_tag = "<This message was edited>"
    parsed_data = []
    available_media_files = []
    if media_folder_path and os.path.isdir(media_folder_path):
        try:
            available_media_files = sorted(os.listdir(media_folder_path))
            available_media_files = [f for f in available_media_files if not f.lower().endswith('.vcf')]
        except OSError as e:
            st.warning(f"Advertencia: No se pudo listar el directorio de medios '{media_folder_path}': {e}")
            media_folder_path = None
    else:
        media_folder_path = None
    used_media_files = set()
    current_message_dict = None
    try:
        with open(txt_filepath, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line: continue
                msg_match = message_pattern.match(line)
                if msg_match:
                    if current_message_dict: parsed_data.append(current_message_dict)
                    date_str, time_str, am_pm, sender, message_content = [g.strip() for g in msg_match.groups()]
                    am_pm = am_pm.upper()
                    try:
                        ts_str_combined = f"{date_str} {time_str} {am_pm}"
                        timestamp = datetime.strptime(ts_str_combined, '%m/%d/%y %I:%M %p')
                    except ValueError:
                        try: timestamp = datetime.strptime(ts_str_combined, '%m/%d/%Y %I:%M %p')
                        except ValueError:
                            st.warning(f"Advertencia: Formato fecha/hora no reconocido línea {line_num}: '{ts_str_combined}'")
                            timestamp = None
                    current_message_dict = {"timestamp": timestamp, "sender": sender, "message": message_content, "is_media": False, "media_type": None, "media_filename": None, "media_filepath": None}
                    media_explicit_match = media_explicit_pattern.match(message_content)
                    if media_explicit_match:
                        filename = media_explicit_match.group(1)
                        if not filename.lower().endswith('.vcf'):
                            current_message_dict.update({"is_media": True, "media_filename": filename, "media_type": filename.split('.')[-1].lower(), "message": ""})
                            if media_folder_path:
                                potential_path = os.path.join(media_folder_path, filename)
                                if os.path.exists(potential_path) and filename not in used_media_files:
                                    current_message_dict["media_filepath"] = potential_path
                                    used_media_files.add(filename)
                    elif message_content == media_omitted_marker:
                        current_message_dict.update({"is_media": True, "media_type": "omitted", "message": ""})
                    if isinstance(current_message_dict["message"], str) and current_message_dict["message"].endswith(edited_tag):
                        current_message_dict["message"] = current_message_dict["message"][:-len(edited_tag)].strip()
                elif current_message_dict:
                    current_message_dict["message"] += "\n" + line
                    if isinstance(current_message_dict["message"], str) and current_message_dict["message"].endswith(edited_tag):
                        current_message_dict["message"] = current_message_dict["message"][:-len(edited_tag)].strip()
                else:
                    parsed_data.append({"timestamp": None, "sender": "System", "message": line, "is_media": False, "media_type": None, "media_filename": None, "media_filepath": None})
                    current_message_dict = None
            if current_message_dict: parsed_data.append(current_message_dict)
    except FileNotFoundError: st.error(f"Error: Archivo no encontrado {txt_filepath}"); return pd.DataFrame()
    except Exception as e: st.error(f"Error procesando TXT: {e}"); import traceback; st.error(traceback.format_exc()); return pd.DataFrame()
    if media_folder_path:
        media_index_sequential = 0
        for msg_data in parsed_data:
            if msg_data.get("is_media") and msg_data.get("media_type") == "omitted" and msg_data.get("media_filepath") is None:
                while media_index_sequential < len(available_media_files) and (available_media_files[media_index_sequential] in used_media_files or available_media_files[media_index_sequential].lower().endswith('.vcf')):
                    media_index_sequential += 1
                if media_index_sequential < len(available_media_files):
                    filename = available_media_files[media_index_sequential]
                    potential_path = os.path.join(media_folder_path, filename)
                    msg_data.update({"media_filepath": potential_path, "media_filename": filename, "media_type": filename.split('.')[-1].lower()})
                    used_media_files.add(filename); media_index_sequential += 1
    df = pd.DataFrame(parsed_data)
    if not df.empty:
        cols_order = ['timestamp', 'sender', 'message', 'is_media', 'media_type', 'media_filename', 'media_filepath']
        df = df[[col for col in cols_order if col in df.columns]]
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
    return df

def configure_gemini_api_st():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        if not api_key or api_key == "TU_API_KEY_VA_AQUI":#SOLO PARA TESTS
            st.error("API Key de Google no configurada correctamente en secrets.toml."); return False
        genai.configure(api_key=api_key); return True
    except KeyError: st.error("GOOGLE_API_KEY no encontrada en st.secrets."); st.info("Crea .streamlit/secrets.toml con tu GOOGLE_API_KEY='tu_clave'"); return False
    except Exception as e: st.error(f"Error CRÍTICO configurando API Gemini: {e}"); return False

def get_text_conversation(df_filtered):
    if 'message' not in df_filtered.columns: return ""
    df_filtered_copy = df_filtered.copy()
    df_filtered_copy['message'] = df_filtered_copy['message'].astype(str)
    text_df = df_filtered_copy[(df_filtered_copy['is_media'] == False) & (df_filtered_copy['sender'] != 'System') & (df_filtered_copy['message'].notna()) & (df_filtered_copy['message'].str.strip() != '')]
    if text_df.empty: return ""
    if 'timestamp' in text_df.columns and pd.api.types.is_datetime64_any_dtype(text_df['timestamp']):
        text_df['timestamp_str'] = text_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        return "\n".join([f"[{row['timestamp_str']}] {row['sender']}: {row['message']}" for _, row in text_df.iterrows()])
    else:
        return "\n".join([f"{row['sender']}: {row['message']}" for _, row in text_df.iterrows()])

def get_context_for_media(df_full, media_index_in_full_df, window=5):
    if media_index_in_full_df is None or media_index_in_full_df not in df_full.index: return "(Contexto no disponible)"
    if 'message' not in df_full.columns: return "(Columna 'message' no encontrada para contexto)"
    df_full_copy = df_full.copy(); df_full_copy['message'] = df_full_copy['message'].astype(str)
    try: media_loc = df_full_copy.index.get_loc(media_index_in_full_df)
    except KeyError: return "(Índice del medio no encontrado)"
    start_idx = max(0, media_loc - window); end_idx = min(len(df_full_copy), media_loc + window + 1)
    context_df_slice = df_full_copy.iloc[start_idx:end_idx]
    if context_df_slice.empty: return "(Contexto vacío)"
    context_lines = []
    for idx, row in context_df_slice.iterrows():
        prefix_parts = []
        if pd.notna(row.get('timestamp')) and pd.api.types.is_datetime64_any_dtype(row['timestamp']): prefix_parts.append(f"[{row['timestamp'].strftime('%Y-%m-%d %H:%M')}]")
        prefix_parts.append(f"{row.get('sender', 'Desconocido')}:")
        prefix = " ".join(prefix_parts)
        if idx == media_index_in_full_df: context_lines.append(f"{prefix} <ARCHIVO MULTIMEDIA ADJUNTO A ANALIZAR>")
        elif not row.get('is_media', False) and pd.notna(row.get('message')) and str(row.get('message')).strip() != '': context_lines.append(f"{prefix} {row['message']}")
        elif row.get('is_media', False) and pd.notna(row.get('media_type')): context_lines.append(f"{prefix} <{row['media_type']} adjunto>")
    return "\n".join(context_lines)

#---- PROMPT para Resumen--- Cambiar si se necesita otro contexto
def run_text_analysis_gemini_and_return(conversation_text, start_date_str, end_date_str, text_model_name):
    if not conversation_text: st.warning("No hay texto para el resumen de Gemini."); return None
    prompt_resumen = f"""Eres un analista experto en grupos de WhatsApp de conductores de plataformas como DiDi en México. Has recibido el siguiente historial de chat de un grupo de conductores de DiDi en Los Mochis correspondiente al periodo entre {start_date_str} y {end_date_str}.
    Por favor, lee cuidadosamente la conversación y genera un resumen conciso (máximo 300 palabras) que incluya:
    1. Los 3-4 temas o problemas principales mencionados por los conductores durante este periodo (ej. tarifas bajas, cupones, competencia, incentivos, seguridad, problemas con la app, horarios, etc.).
    2. El sentimiento general expresado sobre esos temas (frustración, esperanza, resignación, etc.).
    3. Cualquier sugerencia o solución propuesta por los conductores o por el personal de DiDi presente.
    4. Menciona brevemente si hubo discusiones sobre otras plataformas (Uber, InDriver).
    CONVERSACIÓN DEL PERIODO:\n{conversation_text}\n\nRESUMEN DEL PERIODO ({start_date_str} a {end_date_str}):"""
    try:
        model_text = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=8192)
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model_text.generate_content(prompt_resumen, generation_config=generation_config, safety_settings=safety_settings)
        if response.parts: return response.text
        else:
            st.error("Respuesta de Gemini (resumen) bloqueada/vacía.");
            if hasattr(response, 'prompt_feedback'): st.json(response.prompt_feedback)
            if hasattr(response, 'candidates') and response.candidates: st.write("Bloqueo:", response.candidates[0].finish_reason); st.write("Safety Ratings:", response.candidates[0].safety_ratings)
            return None
    except Exception as e: st.error(f"Error en resumen con Gemini ({text_model_name}): {e}"); st.exception(e); return None

#---- Funcion para analisis de imagenes-- Cambiar Prompt para tener otro contexto
#Se añadio una revision para definir si una imagen es relevante

def run_image_analysis_gemini_and_display(df_full_chat, df_filtered_for_selection, max_images_to_analyze,
                                          vision_model_name):
    st.subheader(f"🖼️ Análisis de Imágenes Relevantes con Gemini (primeras {max_images_to_analyze} consideradas)")

    candidate_images = df_filtered_for_selection[
        (df_filtered_for_selection['is_media'] == True) &
        (df_filtered_for_selection['media_type'].isin(['jpg', 'jpeg', 'png', 'webp'])) &
        (df_filtered_for_selection['media_filepath'].notna()) &
        (df_filtered_for_selection['media_filepath'].str.lower().str.contains(r'\.(jpg|jpeg|png|webp)$', regex=True,
                                                                              na=False))
        ].copy()

    if candidate_images.empty:
        st.info(
            "No se encontraron imágenes (JPG, JPEG, PNG, WEBP) con ruta válida en el rango de fechas para análisis con Gemini.")
        return

    st.write(
        f"Se encontraron {len(candidate_images)} imágenes candidatas en el periodo. Pre-filtrando y analizando hasta las primeras {max_images_to_analyze} imágenes relevantes...")

    images_processed_for_relevance = 0
    images_analyzed_in_detail = 0

    try:
        # Se usa el mismo modelo para clasificacion y analisis detallado si es necesario usar un modelo mas sencillo
        model_vision = genai.GenerativeModel(vision_model_name)

        # Configuración de seguridad 
        safety_settings_vision = [{"category": c, "threshold": "BLOCK_NONE"} for c in
                                  ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                   "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        for original_index, row_image in candidate_images.iterrows():
            if images_processed_for_relevance >= (max_images_to_analyze * 2) + 5 and images_analyzed_in_detail >= max_images_to_analyze :
                st.caption(f"Se alcanzó un límite de consideración de imágenes para mantener el rendimiento.")
                break
            if images_analyzed_in_detail >= max_images_to_analyze:
                st.caption(f"Se alcanzó el límite de {max_images_to_analyze} imágenes analizadas en detalle.")
                break

            images_processed_for_relevance += 1
            image_path = row_image['media_filepath']

            if not os.path.exists(image_path):
                st.warning(f"Archivo de imagen no encontrado: {image_path}. Saltando.");
                continue

            try:
                img_pil = Image.open(image_path)
                if img_pil.mode == 'RGBA': img_pil = img_pil.convert('RGB')

                # --- PASO 1: Clasificación de Relevancia ---
                prompt_relevance_check = """Analiza la siguiente imagen. Determina si su contenido visual es **directa y significativamente relevante** para un análisis de negocio o de operaciones de DiDi.
Busca elementos como:
- Capturas de pantalla de la aplicación DiDi (ganancias, mapas, mensajes de la app, problemas técnicos).
- Comunicaciones oficiales de DiDi (banners, anuncios de incentivos, cambios de tarifa).
- Precios, tarifas, o discusiones sobre ingresos claramente visibles.
- Marketing o publicidad de DiDi.
- Vehículos claramente identificados como DiDi en un contexto de trabajo (ej. mostrando logos, en zonas de espera designadas).
- Problemas mecánicos o de seguridad del vehículo si el contexto sugiere que es un vehículo de trabajo.

Si la imagen NO cumple con estos criterios de alta relevancia directa para DiDi (por ejemplo, si es un meme genérico, una foto personal, un paisaje, comida, un vehículo sin clara identificación DiDi o contexto laboral, o contenido ambiguo), clasifícala como 'NO PRIORITARIA'.
Si SÍ cumple con los criterios de alta relevancia directa, responde 'PRIORITARIA'.
Responde únicamente con 'PRIORITARIA' o 'NO PRIORITARIA'.
"""
                is_relevant = False
                with st.spinner(f"Clasificando relevancia de '{row_image['media_filename']}'..."):
                    generation_config_relevance = genai.types.GenerationConfig(temperature=0.1,
                                                                               max_output_tokens=50)
                    try:
                        response_relevance = model_vision.generate_content(
                            [prompt_relevance_check, img_pil],
                            generation_config=generation_config_relevance,
                            safety_settings=safety_settings_vision
                        )
                        if response_relevance.parts and response_relevance.text:
                            classification = response_relevance.text.strip().upper()
                            if "PRIORITARIA" in classification:
                                is_relevant = True
                        else:
                            # Si la respuesta es bloqueada o vacía, se asume no relevante.
                            pass
                    except Exception:
                        # En caso de error en la API de clasificación, asumir no relevante.
                        pass

                if not is_relevant:
                    # st.caption(f"'{row_image['media_filename']}' clasificada como no relevante o error en clasificación.") # Log opcional
                    continue

                images_analyzed_in_detail += 1

                # --- PASO 2: Análisis Detallado (si es relevante) ---
                with st.expander(f"Análisis de Imagen Relevante: {row_image['media_filename']}",
                                 expanded=True):
                    st.image(image_path, width=300)
                    context = get_context_for_media(df_full_chat, original_index, window=4)
                    prompt_vision_detailed = f"""Eres un asistente experto analizando conversaciones de WhatsApp de conductores de DiDi. La siguiente imagen ha sido considerada RELEVANTE para el contexto laboral de DiDi.
                    Observa la imagen adjunta y lee el contexto de la conversación proporcionado (la imagen está indicada con <ARCHIVO MULTIMEDIA ADJUNTO A ANALIZAR>).
                    Basado en ambos (imagen y texto):
                    1. Describe concisamente el contenido principal de la imagen (ej. captura de app de ganancias, foto de vehículo/calle, meme relevante al trabajo, problema mecánico, etc.).
                    2. ¿Cuál es el propósito probable por el que el remitente compartió esta imagen en la conversación, según el contexto?
                    CONTEXTO DE LA CONVERSACIÓN:\n{context}\n\nANÁLISIS DE IMAGEN:"""

                    with st.spinner(
                            f"Gemini ({vision_model_name}) está analizando '{row_image['media_filename']}' en detalle... 🖼️"):
                        generation_config_detailed = genai.types.GenerationConfig(temperature=0.4,
                                                                                  max_output_tokens=2048)
                        response_detailed = model_vision.generate_content(
                            [prompt_vision_detailed, img_pil],
                            generation_config=generation_config_detailed,
                            safety_settings=safety_settings_vision
                        )
                        if response_detailed.parts and response_detailed.text:
                            st.markdown(f"**Análisis de Gemini para {row_image['media_filename']}:**")
                            st.markdown(response_detailed.text)
                        else:
                            st.error(
                                f"Respuesta de Gemini (análisis detallado de {row_image['media_filename']}) bloqueada/vacía.")
                            if hasattr(response_detailed, 'prompt_feedback'): st.json(response_detailed.prompt_feedback)
                            if hasattr(response_detailed, 'candidates') and response_detailed.candidates:
                                st.write("Razón de bloqueo:", response_detailed.candidates[0].finish_reason)
                                st.write("Safety Ratings:", response_detailed.candidates[0].safety_ratings)

                if images_analyzed_in_detail < min(len(candidate_images), max_images_to_analyze):
                    time.sleep(3)

            except FileNotFoundError:
                st.warning(f"Error: No se pudo encontrar/abrir la imagen en la ruta: {image_path}")
            except Exception as e_inner:
                st.error(f"Error durante el procesamiento de la imagen {row_image['media_filename']}: {e_inner}")
                st.exception(e_inner)

        if images_analyzed_in_detail == 0 and images_processed_for_relevance > 0:
            st.info(
                "Se revisaron algunas imágenes, pero ninguna fue considerada suficientemente relevante para un análisis detallado según los criterios.")
        elif images_analyzed_in_detail == 0 and images_processed_for_relevance == 0 and not candidate_images.empty:
            st.info("No se procesaron imágenes para relevancia (podría ser un límite alcanzado o error inicial).")

    except Exception as e_outer:
        st.error(f"Error general durante el análisis multimodal con Gemini ({vision_model_name}): {e_outer}")
        st.info(f"Asegúrate que el modelo '{vision_model_name}' es correcto y tienes acceso a él.")
        st.exception(e_outer)

#----Funcion para Gemini determine principales temas y pain points-- Cambiar prompt para diferente contexto
def get_topics_pain_points_gemini(text_content, text_model_name, start_date_str, end_date_str):
    if not text_content: return "No hay texto para analizar temas."
    prompt = f"""Eres un analista experto de conversaciones de WhatsApp, especializado en identificar problemas y temas recurrentes en grupos de conductores de DiDi en México.
    Analiza la siguiente conversación del periodo {start_date_str} al {end_date_str}:
    CONVERSACIÓN:\n{text_content}\n
    Por favor, identifica y lista los siguientes puntos de forma clara y concisa:
    1.  **Principales Temas de Conversación (3-5 temas):** Menciona los temas más discutidos.
    2.  **Puntos de Dolor o Quejas Comunes (Pain Points) (3-5 puntos):** ¿Cuáles son las frustraciones o problemas más expresados por los conductores?
    3.  **Sugerencias o Soluciones Propuestas (si las hay):** ¿Se mencionaron ideas para mejorar?
    Presenta cada sección claramente. Utiliza viñetas para los puntos dentro de cada sección."""
    try:
        model = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.5, max_output_tokens=8192)
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        if response.parts: return response.text
        else:
            st.error("Respuesta de Gemini (temas/pain points) bloqueada/vacía.")
            if hasattr(response, 'prompt_feedback'): st.json(response.prompt_feedback)
            if hasattr(response, 'candidates') and response.candidates: st.write("Bloqueo:", response.candidates[0].finish_reason); st.write("Safety Ratings:", response.candidates[0].safety_ratings)
            return None
    except Exception as e: st.error(f"Error identificando temas/pain points con Gemini ({text_model_name}): {e}"); st.exception(e); return None

#Funcion para analisis de sentimiento-- Cambiar Prompt para diferente Contexto
def get_overall_sentiment_gemini(text_content, text_model_name, start_date_str, end_date_str):
    if not text_content: return {"category": "N/A", "explanation": "No hay texto para analizar sentimiento."}
    prompt = f"""Eres un experto en análisis de sentimiento. Analiza la siguiente conversación de un grupo de WhatsApp de conductores de DiDi en México, del periodo {start_date_str} al {end_date_str}:
    CONVERSACIÓN:\n{text_content}\n
    Considerando el tono general, las emociones expresadas y la predominancia de comentarios positivos, negativos o neutrales, por favor:
    1.  Clasifica el sentimiento general del periodo en una de las siguientes categorías: **Muy Positivo, Positivo, Neutral, Negativo, Muy Negativo**.
    2.  Proporciona una breve explicación (1-2 frases) que justifique tu clasificación.
    RESPUESTA ESPERADA (ejemplo):\nCategoría de Sentimiento: Neutral\nExplicación: Hubo una mezcla de quejas sobre tarifas y algunos comentarios positivos sobre nuevos incentivos."""
    try:
        model = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.3, max_output_tokens=8192)
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        if response.parts and response.text:
            category = "No determinado"; explanation = response.text
            lines = response.text.split('\n')
            for line in lines:
                if "Categoría de Sentimiento:" in line: category = line.split("Categoría de Sentimiento:")[1].strip()
                elif "Explicación:" in line: explanation = line.split("Explicación:")[1].strip()
            if explanation == response.text and category != "No determinado":
                parts = response.text.split("Explicación:")
                if len(parts) > 1: explanation = parts[1].strip()
                else:
                    parts_after_category = response.text.split(category)
                    if len(parts_after_category) > 1: explanation = parts_after_category[1].strip()
            return {"category": category, "explanation": explanation}
        else:
            st.error("Respuesta de Gemini (sentimiento) bloqueada/vacía.")
            if hasattr(response, 'prompt_feedback'): st.json(response.prompt_feedback)
            if hasattr(response, 'candidates') and response.candidates: st.write("Bloqueo:", response.candidates[0].finish_reason); st.write("Safety Ratings:", response.candidates[0].safety_ratings)
            return {"category": "Error", "explanation": "Respuesta bloqueada o vacía."}
    except Exception as e: st.error(f"Error evaluando sentimiento con Gemini ({text_model_name}): {e}"); st.exception(e); return {"category": "Error", "explanation": str(e)}

# --- Dashboard ---
st.set_page_config(layout="wide", page_title="Analizador de Chats de WhatsApp con IA")
st.title(" Analizador de Grupos de WhatsApp con IA 🧠")
st.markdown("Sube el archivo `.txt` de tu chat de WhatsApp y, opcionalmente, los archivos multimedia asociados para un análisis detallado.")

gemini_api_configured = configure_gemini_api_st()

if 'df_chat_full' not in st.session_state: st.session_state.df_chat_full = pd.DataFrame()
if 'df_chat_filtered' not in st.session_state: st.session_state.df_chat_filtered = pd.DataFrame()
if 'temp_dir' not in st.session_state: st.session_state.temp_dir = None
if 'start_date_filter' not in st.session_state: st.session_state.start_date_filter = date.today() - timedelta(days=7)
if 'end_date_filter' not in st.session_state: st.session_state.end_date_filter = date.today()
if 'gemini_summary' not in st.session_state: st.session_state.gemini_summary = None
if 'gemini_topics_pain_points' not in st.session_state: st.session_state.gemini_topics_pain_points = None
if 'gemini_sentiment_score' not in st.session_state: st.session_state.gemini_sentiment_score = None

st.sidebar.header("📂 Cargar Archivos")
uploaded_chat_file = st.sidebar.file_uploader("1. Sube tu archivo de chat (.txt)", type="txt")
uploaded_media_files = st.sidebar.file_uploader("2. (Opcional) Sube archivos multimedia", accept_multiple_files=True)

if uploaded_chat_file is not None:
    if st.sidebar.button("🚀 Procesar Chat", type="primary"):
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            try: shutil.rmtree(st.session_state.temp_dir)
            except Exception as e: st.warning(f"No se pudo limpiar dir temporal anterior: {e}")
        st.session_state.temp_dir = tempfile.mkdtemp()
        temp_chat_path = os.path.join(st.session_state.temp_dir, uploaded_chat_file.name)
        with open(temp_chat_path, "wb") as f: f.write(uploaded_chat_file.getbuffer())
        temp_media_folder_path = None
        if uploaded_media_files:
            temp_media_folder_path = os.path.join(st.session_state.temp_dir, "media")
            os.makedirs(temp_media_folder_path, exist_ok=True)
            for uploaded_file_media in uploaded_media_files:
                with open(os.path.join(temp_media_folder_path, uploaded_file_media.name), "wb") as f:
                    f.write(uploaded_file_media.getbuffer())
        with st.spinner("Analizando el chat... Esto puede tardar unos momentos. ⏳"):
            try:
                df_full = parse_whatsapp_chat_final(temp_chat_path, temp_media_folder_path)
                st.session_state.df_chat_full = df_full
                if not df_full.empty and 'timestamp' in df_full.columns and not df_full['timestamp'].isnull().all():
                    st.session_state.start_date_filter = df_full['timestamp'].min().date()
                    st.session_state.end_date_filter = df_full['timestamp'].max().date()
                else:
                    st.session_state.start_date_filter = date.today() - timedelta(days=7)
                    st.session_state.end_date_filter = date.today()
                st.session_state.df_chat_filtered = df_full
                st.session_state.gemini_summary = None
                st.session_state.gemini_topics_pain_points = None
                st.session_state.gemini_sentiment_score = None
                if 'run_gemini_analysis' in st.session_state: del st.session_state.run_gemini_analysis
                if not st.session_state.df_chat_full.empty:
                    st.success(f"¡Chat procesado! Se encontraron {len(st.session_state.df_chat_full)} mensajes/líneas.")
                else: st.warning("El chat fue procesado, pero no se extrajeron datos.")
            except Exception as e:
                st.error(f"Ocurrió un error crítico durante el parseo: {e}"); st.exception(e)
                st.session_state.df_chat_full = pd.DataFrame(); st.session_state.df_chat_filtered = pd.DataFrame()
else:
    if st.session_state.df_chat_full.empty:
        st.info("👈 Sube un archivo de chat y presiona 'Procesar Chat' para comenzar.")

if not st.session_state.df_chat_full.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("🗓️ Filtrar por Fecha (para vista previa y análisis)")
    df_full_for_dates = st.session_state.df_chat_full
    min_date_chat = df_full_for_dates['timestamp'].min().date() if not df_full_for_dates.empty and 'timestamp' in df_full_for_dates.columns and not df_full_for_dates['timestamp'].isnull().all() else st.session_state.start_date_filter
    max_date_chat = df_full_for_dates['timestamp'].max().date() if not df_full_for_dates.empty and 'timestamp' in df_full_for_dates.columns and not df_full_for_dates['timestamp'].isnull().all() else st.session_state.end_date_filter
    start_date_val = st.session_state.start_date_filter
    if not (min_date_chat <= start_date_val <= max_date_chat): start_date_val = min_date_chat
    end_date_val = st.session_state.end_date_filter
    if not (min_date_chat <= end_date_val <= max_date_chat): end_date_val = max_date_chat
    if start_date_val > end_date_val: start_date_val = end_date_val
    date_range = st.sidebar.date_input(
        "Selecciona el rango de fechas:", value=(start_date_val, end_date_val),
        min_value=min_date_chat, max_value=max_date_chat, key="date_range_picker"
    )
    if date_range and len(date_range) == 2:
        start_date_selected, end_date_selected = date_range
        if start_date_selected > end_date_selected: st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            st.session_state.start_date_filter = start_date_selected
            st.session_state.end_date_filter = end_date_selected
            start_datetime = datetime.combine(start_date_selected, datetime.min.time())
            end_datetime = datetime.combine(end_date_selected, datetime.max.time())
            mask = (df_full_for_dates['timestamp'] >= start_datetime) & (df_full_for_dates['timestamp'] <= end_datetime)
            st.session_state.df_chat_filtered = df_full_for_dates.loc[mask]
    else: st.session_state.df_chat_filtered = df_full_for_dates

if not st.session_state.df_chat_filtered.empty and gemini_api_configured:
    st.sidebar.markdown("---"); st.sidebar.header("🧠 Análisis con Gemini IA")
    max_images_gemini_input = st.sidebar.number_input("Máx. imágenes a analizar con IA:", min_value=0, max_value=10, value=3, step=1, help="0 para no analizar imágenes.")
    if st.sidebar.button("✨ Ejecutar Análisis con Gemini", type="primary", help="Realiza el análisis sobre el periodo filtrado."):
        st.session_state['run_gemini_analysis'] = True
        st.session_state['max_images_gemini_run'] = max_images_gemini_input
        st.session_state.gemini_summary = None
        st.session_state.gemini_topics_pain_points = None
        st.session_state.gemini_sentiment_score = None
elif not st.session_state.df_chat_filtered.empty and not gemini_api_configured:
    st.sidebar.markdown("---"); st.sidebar.warning("API de Gemini no configurada. Análisis con IA no disponible.")

if 'run_gemini_analysis' in st.session_state and st.session_state.run_gemini_analysis:
    if not st.session_state.df_chat_filtered.empty and gemini_api_configured:
        df_filtered_for_analysis = st.session_state.df_chat_filtered
        start_date_str_gemini = st.session_state.start_date_filter.strftime('%Y-%m-%d')
        end_date_str_gemini = st.session_state.end_date_filter.strftime('%Y-%m-%d')
        conversation_text_for_analysis = get_text_conversation(df_filtered_for_analysis)
        if st.session_state.gemini_summary is None:
            st.session_state.gemini_summary = run_text_analysis_gemini_and_return(conversation_text_for_analysis, start_date_str_gemini, end_date_str_gemini, TEXT_MODEL_NAME_GEMINI)
        if st.session_state.gemini_topics_pain_points is None:
            st.session_state.gemini_topics_pain_points = get_topics_pain_points_gemini(conversation_text_for_analysis, TEXT_MODEL_NAME_GEMINI, start_date_str_gemini, end_date_str_gemini)
        if st.session_state.gemini_sentiment_score is None:
            st.session_state.gemini_sentiment_score = get_overall_sentiment_gemini(conversation_text_for_analysis, TEXT_MODEL_NAME_GEMINI, start_date_str_gemini, end_date_str_gemini)
    del st.session_state.run_gemini_analysis

if not st.session_state.df_chat_filtered.empty:
    tab1, tab2 = st.tabs(["🤖 Resumen General y Multimedia (IA)", "📊 Estadísticas y Sentimiento Detallado"])
    with tab1:
        st.header("🤖 Resumen General y Multimedia (IA)")
        if gemini_api_configured:
            if st.session_state.gemini_summary:
                st.subheader("💬 Resumen del Periodo por Gemini")
                st.markdown(st.session_state.gemini_summary)
            elif uploaded_chat_file and st.session_state.get('gemini_summary') is None:
                st.info("Presiona 'Ejecutar Análisis con Gemini' en la barra lateral para ver el resumen y análisis de imágenes.")
            max_images_to_run_val_tab1 = st.session_state.get('max_images_gemini_run')
            if max_images_to_run_val_tab1 is not None and max_images_to_run_val_tab1 > 0 :
                run_image_analysis_gemini_and_display(st.session_state.df_chat_full, st.session_state.df_chat_filtered, max_images_to_run_val_tab1, VISION_MODEL_NAME_GEMINI)
                if 'max_images_gemini_run' in st.session_state: del st.session_state.max_images_gemini_run
            elif max_images_to_run_val_tab1 == 0:
                st.info("Análisis de imágenes con IA desactivado (máximo de imágenes es 0).")
        else: st.warning("API de Gemini no configurada. Análisis con IA no disponible.")
        with st.expander("📂 Vista Previa de los Datos del Chat (Filtrados/Completos)", expanded=False):
            df_display = st.session_state.df_chat_filtered
            df_full_display = st.session_state.df_chat_full
            if not df_display.empty:
                if len(df_display) < len(df_full_display) and not df_full_display.empty :
                    st.caption(f"Mostrando {len(df_display)} de {len(df_full_display)} mensajes totales, según el filtro de fecha.")
                elif not df_display.empty: st.caption(f"Mostrando todos los {len(df_display)} mensajes procesados.")
                st.dataframe(df_display.head(10))
                @st.cache_data
                def convert_df_to_csv(df_to_convert): return df_to_convert.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                csv_data_filtered = convert_df_to_csv(df_display)
                file_suffix = "filtrado" if not df_full_display.empty and len(df_display) < len(df_full_display) else "completo"
                st.download_button(label=f"📥 Descargar datos ({file_suffix}) como CSV", data=csv_data_filtered, file_name=f"chat_parseado_{file_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv')
            else: st.caption("No hay datos para mostrar en la vista previa (posiblemente debido al filtro).")

    with tab2:
        st.header("📊 Estadísticas Detalladas y Sentimiento")
        df_stats = st.session_state.df_chat_filtered.copy()
        st.subheader("🔢 Métricas Clave del Periodo")
        total_mensajes_filtrados = len(df_stats)
        mensajes_multimedia_filtrados = df_stats['is_media'].sum() if 'is_media' in df_stats else 0
        participantes_unicos = df_stats[df_stats['sender'] != 'System']['sender'].nunique() if 'sender' in df_stats.columns else 0
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1: st.metric(label="Total de Mensajes", value=total_mensajes_filtrados)
        with mcol2: st.metric(label="Mensajes Multimedia", value=mensajes_multimedia_filtrados)
        with mcol3: st.metric(label="Participantes Únicos", value=participantes_unicos)
        st.markdown("---")
        if 'sender' in df_stats.columns:
            st.subheader("🗣️ Usuarios Más Activos")
            mensajes_por_usuario = df_stats[df_stats['sender'] != 'System']['sender'].value_counts().head(10)
            if not mensajes_por_usuario.empty: st.bar_chart(mensajes_por_usuario, use_container_width=True)
            else: st.info("No hay mensajes de usuarios para mostrar actividad.")
        st.markdown("---")
        if 'timestamp' in df_stats.columns:
            st.subheader("📈 Actividad por Día")
            df_stats_tiempo = df_stats.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_stats_tiempo['timestamp']):
                df_stats_tiempo['timestamp'] = pd.to_datetime(df_stats_tiempo['timestamp'], errors='coerce')
            df_stats_tiempo.dropna(subset=['timestamp'], inplace=True)
            if not df_stats_tiempo.empty:
                mensajes_por_dia = df_stats_tiempo.set_index('timestamp').resample('D')['message'].count()
                if not mensajes_por_dia.empty:
                    st.line_chart(mensajes_por_dia, use_container_width=True)
                    st.markdown("🔥 Top Días de Mayor Actividad")
                    top_dias = mensajes_por_dia.sort_values(ascending=False).head(5)
                    if not top_dias.empty:
                        top_dias.index = top_dias.index.strftime('%Y-%m-%d (%A)')
                        st.bar_chart(top_dias, use_container_width=True)
                    else: st.caption("No hay datos suficientes para mostrar el top de días.")
                else: st.info("No hay datos de mensajes para mostrar la actividad por día.")
            else: st.info("No hay timestamps válidos para mostrar la actividad por día.")
        st.markdown("---")
        if gemini_api_configured:
            st.subheader("🎯 Pain Points (IA)")
            if st.session_state.gemini_topics_pain_points: st.markdown(st.session_state.gemini_topics_pain_points)
            elif uploaded_chat_file: st.info("Presiona 'Ejecutar Análisis con Gemini' para identificar temas y puntos de dolor.")
            st.markdown("---")
            st.subheader("🧐 Medidor de Sentimiento General (IA)")
            if st.session_state.gemini_sentiment_score:
                sentiment_data = st.session_state.gemini_sentiment_score
                sentiment_category = sentiment_data.get("category", "No determinado")
                sentiment_explanation = sentiment_data.get("explanation", "No disponible.")
                delta_text = ""
                if sentiment_category == "Muy Positivo": delta_text = "😃 Muy Bueno"
                elif sentiment_category == "Positivo": delta_text = "🙂 Bueno"
                elif sentiment_category == "Neutral": delta_text = "😐 Neutral"
                elif sentiment_category == "Negativo": delta_text = "🙁 Malo"
                elif sentiment_category == "Muy Negativo": delta_text = "😠 Muy Malo"
                st.metric(label="Sentimiento General", value=sentiment_category, delta=delta_text, delta_color="off" if sentiment_category in ["Neutral", "No determinado"] else ("normal" if sentiment_category in ["Positivo", "Muy Positivo"] else "inverse"))
                st.caption(f"Explicación de Gemini: {sentiment_explanation}")
            elif uploaded_chat_file: st.info("Presiona 'Ejecutar Análisis con Gemini' para ver el análisis de sentimiento.")
        else: st.info("Análisis de temas y sentimiento con IA requiere API de Gemini configurada.")
elif not st.session_state.df_chat_full.empty and st.session_state.df_chat_filtered.empty:
    st.markdown("---"); st.warning("No se encontraron mensajes en el rango de fechas seleccionado. Ajusta el filtro de fechas.")
# --- FIN DE LA APP STREAMLIT ---

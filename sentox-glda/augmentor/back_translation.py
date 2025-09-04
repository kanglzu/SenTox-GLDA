from pathlib import Path
from tqdm import tqdm
import warnings
import time
import re
import jieba
import jieba.posseg as pseg
from google.cloud import translate_v2 as translate
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class BackTranslationAugmentor:
    """Enhanced back-translation with Google Cloud Translation API"""

    def __init__(self, translation_path: str = "zh->ja->en->zh", api_key: Optional[str] = None):
        # Initialize Google Cloud Translation client
        if api_key:
            # If API key is provided, set it as environment variable
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_key

        self.client = translate.Client()
        self.translation_path = translation_path
        self.request_count = 0
        self.rate_limit_delay = 0.1  # 100ms between requests

        # Parse the translation path
        self.language_sequence = self._parse_translation_path(translation_path)
        logger.info(f"Back-translation initialized with path: {translation_path}")

    def _parse_translation_path(self, path: str) -> List[str]:
        """Parse translation path string into language sequence"""
        # Remove spaces and split by ->
        path_clean = path.replace(" ", "")
        languages = path_clean.split("->")

        # Validate and convert language codes for Google Cloud Translation API
        lang_map = {
            'zh-cn': 'zh', 'zh': 'zh', 'chinese': 'zh', '中文': 'zh',
            'en': 'en', 'english': 'en', '英文': 'en',
            'ja': 'ja', 'japanese': 'ja', '日文': 'ja',
            'ko': 'ko', 'korean': 'ko', '韩文': 'ko',
            'fr': 'fr', 'french': 'fr', '法文': 'fr',
            'de': 'de', 'german': 'de', '德文': 'de',
            'es': 'es', 'spanish': 'es', '西班牙文': 'es',
            'ru': 'ru', 'russian': 'ru', '俄文': 'ru'
        }

        processed_langs = []
        for lang in languages:
            lang_lower = lang.lower()
            if lang_lower in lang_map:
                processed_langs.append(lang_map[lang_lower])
            else:
                logger.warning(f"Unknown language code: {lang}, using 'en' as fallback")
                processed_langs.append('en')

        # Ensure path starts and ends with Chinese
        if processed_langs[0] != 'zh':
            processed_langs.insert(0, 'zh')
        if processed_langs[-1] != 'zh':
            processed_langs.append('zh')

        # Remove intermediate 'zh' to avoid unnecessary translations
        final_sequence = [processed_langs[0]]  # Start with zh
        for i in range(1, len(processed_langs) - 1):
            if processed_langs[i] != 'zh':
                final_sequence.append(processed_langs[i])
        final_sequence.append(processed_langs[-1])  # End with zh

        return final_sequence[1:-1]  # Remove start and end 'zh' for processing

    def _rate_limited_translate(self, text: str, target_lang: str, source_lang: str = None) -> str:
        """Rate-limited translation with Google Cloud Translation API"""
        try:
            self.request_count += 1

            # Progressive rate limiting
            if self.request_count % 100 == 0:
                time.sleep(2.0)  # Longer pause every 100 requests
            elif self.request_count % 50 == 0:
                time.sleep(1.0)  # Medium pause every 50 requests
            else:
                time.sleep(self.rate_limit_delay)  # Standard rate limiting

            # Use Google Cloud Translation API
            result = self.client.translate(
                text,
                target_language=target_lang,
                source_language=source_lang
            )

            return result['translatedText'] if result and 'translatedText' in result else text

        except Exception as e:
            logger.warning(f"Translation failed from {source_lang} to {target_lang}: {e}")
            return text

    def augment(self, text: str) -> str:
        """Perform back-translation following configured path"""
        if len(text.strip()) < 3:
            return text

        try:
            current_text = text
            current_lang = 'zh'

            # Forward translation through the sequence
            for target_lang in self.language_sequence:
                translated = self._rate_limited_translate(current_text, target_lang, current_lang)
                if translated == current_text:  # Translation failed
                    logger.warning(f"Translation failed at step {current_lang} -> {target_lang}")
                    return text
                current_text = translated
                current_lang = target_lang

            # Final translation back to Chinese
            final_text = self._rate_limited_translate(current_text, 'zh', current_lang)

            return final_text if final_text.strip() and final_text != text else text

        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text

    def batch_translate(self, texts: List[str], target_lang: str, source_lang: str = None) -> List[str]:
        """Batch translation for better efficiency"""
        try:
            if not texts:
                return []

            # Filter out empty texts
            valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
            if not valid_texts:
                return texts

            indices, text_list = zip(*valid_texts)

            # Use batch translation
            results = self.client.translate(
                text_list,
                target_language=target_lang,
                source_language=source_lang
            )

            # Reconstruct full results list
            translated_texts = texts.copy()
            for idx, result in zip(indices, results):
                if 'translatedText' in result:
                    translated_texts[idx] = result['translatedText']

            return translated_texts

        except Exception as e:
            logger.warning(f"Batch translation failed: {e}")
            return texts

    def set_translation_path(self, new_path: str):
        """Update translation path during runtime"""
        self.translation_path = new_path
        self.language_sequence = self._parse_translation_path(new_path)
        logger.info(f"Translation path updated to: {new_path}")

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages from Google Translate"""
        try:
            languages = self.client.get_languages()
            return [lang['language'] for lang in languages]
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return []

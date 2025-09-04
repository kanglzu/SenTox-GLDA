import argparse
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import re
import jieba
import jieba.posseg as pseg
from augmentor.emoji import EmojiAugmentor
from augmentor.structural import StructuralPerturbationAugmentor
from augmentor.back_translation import BackTranslationAugmentor

warnings.filterwarnings('ignore')

try:
    from nlpcda import Similarword, Homophone, RandomDeleteChar
    from googletrans import Translator
except ImportError:
    print("Please install required packages: pip install nlpcda googletrans==4.0.0rc1 jieba")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TwoStageDataAugmentor:
    """Two-stage augmentation system as per requirements"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._init_augmentors()

        if config.get('random_seed'):
            random.seed(config['random_seed'])
            np.random.seed(config['random_seed'])

    def _init_augmentors(self):
        """Initialize all augmentation methods"""
        # Primary augmentation methods
        self.synonym_aug = Similarword(create_num=1, change_rate=self.config['synonym_rate'])
        self.homophone_aug = Homophone(create_num=1, change_rate=self.config['homophone_rate'])
        self.random_delete_aug = RandomDeleteChar(create_num=1, change_rate=self.config['random_delete_rate'])
        self.structural_aug = StructuralPerturbationAugmentor(self.config['structural_perturbation_rate'])

        # Initialize back-translation with custom path
        self.back_translation_aug = BackTranslationAugmentor(
            translation_path=self.config.get('translation_path', 'zh->ja->en->zh')
        )

        # Emoji augmentor for final stage
        self.emoji_aug = EmojiAugmentor(intensity=self.config['emoji_intensity'])
        logger.info("Two-stage augmentation system initialized")

    def _apply_primary_augmentation(self, text: str, method: str) -> str:
        """Apply single primary augmentation method"""
        try:
            if method == 'synonym':
                results = self.synonym_aug.replace(text)
                return results[0] if results else text
            elif method == 'homophone':
                results = self.homophone_aug.replace(text)
                return results[0] if results else text
            elif method == 'random_delete':
                results = self.random_delete_aug.replace(text)
                return results[0] if results else text
            elif method == 'structural_perturbation':
                return self.structural_aug.augment(text)
            elif method == 'back_translation':
                return self.back_translation_aug.augment(text)
            else:
                return text
        except Exception as e:
            logger.warning(f"Primary augmentation {method} failed: {e}")
            return text

    def _select_secondary_candidates(self, primary_augmented: Dict[str, str], num_secondary: int) -> List[str]:
        """Select candidates for secondary augmentation"""
        available_texts = [text for text in primary_augmented.values() if text.strip()]
        if len(available_texts) <= num_secondary:
            return available_texts
        return random.sample(available_texts, num_secondary)

    def augment_single_text(self, original_text: str) -> List[str]:
        """
        Two-stage augmentation for a single text:
        Stage 1: Mandatory augmentation by each enabled method
        Stage 2: Random secondary augmentation on selected candidates
        Final: Emoji augmentation on all variants
        """
        results = []
        original_text = original_text.strip()

        if len(original_text) < 2:
            return [original_text]

        # Always keep original (will be emoji-augmented later)
        if self.config['keep_original']:
            results.append(original_text)

        # Stage 1: Mandatory primary augmentation
        primary_methods = []
        primary_augmented = {}

        if self.config['use_synonym']:
            primary_methods.append('synonym')
        if self.config['use_homophone']:
            primary_methods.append('homophone')
        if self.config['use_random_delete']:
            primary_methods.append('random_delete')
        if self.config['use_structural_perturbation']:
            primary_methods.append('structural_perturbation')
        if self.config['use_back_translation']:
            primary_methods.append('back_translation')

        # Apply each primary method exactly once
        for method in primary_methods:
            augmented_text = self._apply_primary_augmentation(original_text, method)
            if augmented_text.strip() and augmented_text != original_text:
                primary_augmented[method] = augmented_text
                results.append(augmented_text)

        # Stage 2: Secondary augmentation
        if self.config['num_secondary_augmentations'] > 0 and primary_augmented:
            secondary_candidates = self._select_secondary_candidates(
                primary_augmented,
                self.config['num_secondary_augmentations']
            )

            for candidate in secondary_candidates:
                # Apply random secondary augmentation
                secondary_method = random.choice(primary_methods)
                secondary_augmented = self._apply_primary_augmentation(candidate, secondary_method)

                if secondary_augmented.strip() and secondary_augmented not in results:
                    results.append(secondary_augmented)

        # Final Stage: Apply emoji augmentation to ALL variants
        emoji_augmented_results = []
        for text in results:
            emoji_text = self.emoji_aug.augment(text)
            emoji_augmented_results.append(emoji_text)

        # Remove duplicates while preserving order
        final_results = []
        seen = set()
        for text in emoji_augmented_results:
            if text not in seen:
                final_results.append(text)
                seen.add(text)

        return final_results

    def augment_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Apply two-stage augmentation to entire dataframe"""
        augmented_rows = []

        logger.info(f"Starting two-stage augmentation on {len(df)} samples")
        logger.info(f"Primary methods: {[k for k, v in self.config.items() if k.startswith('use_') and v]}")
        logger.info(f"Secondary augmentations per text: {self.config['num_secondary_augmentations']}")
        logger.info(f"Emoji intensity: {self.config['emoji_intensity']}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Two-stage augmentation"):
            original_text = str(row[text_column])

            try:
                # Generate all augmented variants
                augmented_texts = self.augment_single_text(original_text)

                # Create new rows for each variant
                for aug_text in augmented_texts:
                    new_row = row.copy()
                    new_row[text_column] = aug_text
                    augmented_rows.append(new_row)

            except Exception as e:
                logger.warning(f"Augmentation failed for row {idx}: {e}")
                # Keep original row on failure
                augmented_rows.append(row.copy())

        result_df = pd.DataFrame(augmented_rows)
        result_df.reset_index(drop=True, inplace=True)

        logger.info("Two-stage augmentation completed")
        return result_df


def load_config_from_args(args) -> Dict[str, Any]:
    """Load configuration from command line arguments"""
    config = {
        'input_file': args.input_file,
        'output_file': args.output_file,
        'text_column': args.text_column,

        'use_synonym': args.use_synonym,
        'use_homophone': args.use_homophone,
        'use_random_delete': args.use_random_delete,
        'use_structural_perturbation': args.use_structural_perturbation,
        'use_back_translation': args.use_back_translation,

        'synonym_rate': args.synonym_rate,
        'homophone_rate': args.homophone_rate,
        'random_delete_rate': args.random_delete_rate,
        'structural_perturbation_rate': args.structural_perturbation_rate,
        'translation_path': args.translation_path,  # New parameter

        'num_secondary_augmentations': args.num_secondary_augmentations,
        'emoji_intensity': args.emoji_intensity,

        'keep_original': args.keep_original,
        'random_seed': args.random_seed,
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage SenTox Data Augmentation with Configurable Translation Path")

    # File parameters
    parser.add_argument('--input_file', type=str, required=True, help='Input file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')
    parser.add_argument('--text_column', type=str, required=True, help='Text column name to augment')

    # Primary augmentation methods
    parser.add_argument('--use_synonym', action='store_true', default=True, help='Use synonym replacement')
    parser.add_argument('--use_homophone', action='store_true', default=True, help='Use homophone replacement')
    parser.add_argument('--use_random_delete', action='store_true', default=False, help='Use random deletion')
    parser.add_argument('--use_structural_perturbation', action='store_true', default=True,
                        help='Use structural perturbation')
    parser.add_argument('--use_back_translation', action='store_true', default=False, help='Use back-translation')

    # Augmentation parameters
    parser.add_argument('--synonym_rate', type=float, default=0.3, help='Synonym replacement rate')
    parser.add_argument('--homophone_rate', type=float, default=0.3, help='Homophone replacement rate')
    parser.add_argument('--random_delete_rate', type=float, default=0.1, help='Random deletion rate')
    parser.add_argument('--structural_perturbation_rate', type=float, default=0.3, help='Structural perturbation rate')

    # Back-translation configuration
    parser.add_argument('--translation_path', type=str, default='zh->ja->en->zh',
                        help='Translation path for back-translation (default: zh->ja->en->zh). '
                             'Supported languages: zh/chinese, en/english, ja/japanese, ko/korean, '
                             'fr/french, de/german, es/spanish, ru/russian. '
                             'Example: zh->en->fr->zh or chinese->english->japanese->chinese')

    # Secondary augmentation and emoji parameters
    parser.add_argument('--num_secondary_augmentations', type=int, default=2,
                        help='Number of secondary augmentations to perform')
    parser.add_argument('--emoji_intensity', type=str, default='light',
                        choices=['light', 'deep', 'mixed'], help='Emoji augmentation intensity')

    # Other parameters
    parser.add_argument('--keep_original', action='store_true', default=True, help='Keep original texts')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if not Path(args.input_file).exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        return

    # Validate translation path format
    if args.use_back_translation:
        valid_langs = ['zh', 'chinese', '中文', 'en', 'english', '英文', 'ja', 'japanese', '日文',
                       'ko', 'korean', '韩文', 'fr', 'french', '法文', 'de', 'german', '德文',
                       'es', 'spanish', '西班牙文', 'ru', 'russian', '俄文']

        path_parts = args.translation_path.replace(' ', '').split('->')
        invalid_langs = [lang for lang in path_parts if lang.lower() not in valid_langs]

        if invalid_langs:
            logger.warning(f"Invalid language codes in translation path: {invalid_langs}")
            logger.info(
                "Valid language codes: zh/chinese, en/english, ja/japanese, ko/korean, fr/french, de/german, es/spanish, ru/russian")

        if len(path_parts) < 3:
            logger.warning("Translation path should have at least 3 languages for effective back-translation")

    # Load data
    logger.info(f"Loading data from: {args.input_file}")
    try:
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
        elif args.input_file.endswith('.json'):
            df = pd.read_json(args.input_file)
        elif args.input_file.endswith('.xlsx'):
            df = pd.read_excel(args.input_file)
        else:
            logger.error("Supported formats: .csv, .json, .xlsx")
            return
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return

    if args.text_column not in df.columns:
        logger.error(f"Column '{args.text_column}' not found")
        logger.info(f"Available columns: {list(df.columns)}")
        return

    config = load_config_from_args(args)

    logger.info("Two-stage augmentation configuration:")
    for key, value in config.items():
        if key not in ['input_file', 'output_file']:
            logger.info(f"  {key}: {value}")

    # Special log for translation path
    if args.use_back_translation:
        logger.info(f"Back-translation will follow path: {args.translation_path}")

    try:
        augmentor = TwoStageDataAugmentor(config)
        augmented_df = augmentor.augment_dataframe(df, args.text_column)

        # Save results
        logger.info(f"Saving results to: {args.output_file}")
        if args.output_file.endswith('.csv'):
            augmented_df.to_csv(args.output_file, index=False, encoding='utf-8')
        elif args.output_file.endswith('.json'):
            augmented_df.to_json(args.output_file, orient='records', ensure_ascii=False, indent=2)
        elif args.output_file.endswith('.xlsx'):
            augmented_df.to_excel(args.output_file, index=False)
        else:
            augmented_df.to_csv(args.output_file + '.csv', index=False, encoding='utf-8')

        logger.info("Two-stage data augmentation completed successfully!")

        # Statistics
        logger.info("=== Augmentation Statistics ===")
        logger.info(f"Original samples: {len(df)}")
        logger.info(f"Final samples: {len(augmented_df)}")
        logger.info(f"Augmentation ratio: {len(augmented_df) / len(df):.2f}x")

        # Calculate expected ratio
        enabled_methods = sum([
            args.use_synonym, args.use_homophone, args.use_random_delete,
            args.use_structural_perturbation, args.use_back_translation
        ])
        expected_base = 1 + enabled_methods if args.keep_original else enabled_methods
        expected_with_secondary = expected_base + args.num_secondary_augmentations
        logger.info(f"Expected ratio (before emoji): ~{expected_with_secondary}x")

        if args.use_back_translation:
            logger.info(f"Back-translation used path: {args.translation_path}")

    except Exception as e:
        logger.error(f"Error during two-stage augmentation: {e}")
        raise


if __name__ == "__main__":
    main()

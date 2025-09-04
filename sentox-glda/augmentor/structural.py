import logging
from pathlib import Path
from random import random
from typing import List, Tuple
from tqdm import tqdm
import warnings
import time
import re
import jieba
import jieba.posseg as pseg
import logging

logger = logging.getLogger(__name__)

class StructuralPerturbationAugmentor:
    """Comprehensive structural perturbation with deep linguistic patterns"""

    def __init__(self, perturbation_rate: float = 0.3):
        self.perturbation_rate = perturbation_rate

        # Discourse particles by position and function
        self.discourse_particles = {
            'sentence_end': ["啊", "呢", "吧", "哦", "嗯", "呀", "咯", "哈", "喔", "额", "诶", "嘛", "呗", "哇", "嘞",
                             "咧", "噢", "嗳"],
            'sentence_start': ["哎", "唉", "咦", "嘿", "喂", "诶", "嗯", "哦", "呃", "额", "这个", "那个"],
            'mid_sentence': ["呀", "啊", "呢", "嘛", "咯", "喔", "哈", "嘞", "诶", "额"]
        }

        # Intensifiers by semantic category
        self.intensifiers = {
            'degree_high': ["很", "非常", "特别", "超", "极其", "相当", "挺", "蛮", "真", "超级", "巨", "老", "贼",
                            "死", "可", "怪", "好"],
            'degree_extreme': ["极度", "无比", "异常", "格外", "十分", "万分", "极为", "颇为", "尤为", "甚为"],
            'emphasis': ["就是", "正是", "恰恰", "偏偏", "竟然", "居然", "简直", "完全", "绝对", "彻底"],
            'negative': ["一点也不", "根本不", "完全不", "绝对不", "从不", "决不", "毫不", "丝毫不"]
        }

        # Connectors by logical relationship
        self.connectors = {
            'temporal': ["然后", "接着", "于是", "随后", "之后", "后来", "接下来", "紧接着", "马上", "立刻", "顿时",
                         "瞬间"],
            'causal': ["所以", "因此", "于是", "结果", "那么", "这样", "由此", "从而", "因而", "故而", "为此"],
            'adversative': ["但是", "不过", "然而", "可是", "只是", "却", "而", "倒是", "反而", "相反", "与此相反"],
            'additive': ["而且", "还有", "另外", "此外", "再说", "再者", "况且", "何况", "不仅", "不但", "同时"],
            'conditional': ["如果", "要是", "假如", "倘若", "万一", "一旦", "只要", "除非", "无论", "不管"]
        }

        # Modal and attitudinal particles
        self.modal_particles = {
            'uncertainty': ["吧", "呢", "吗", "么", "嘛"],
            'confirmation': ["啊", "呀", "哦", "噢", "唉"],
            'surprise': ["咦", "哎", "呀", "哇", "天哪", "我去"],
            'emphasis': ["啊", "呀", "呢", "嘛", "咯", "喔"]
        }

        # Repetition and emphasis patterns
        self.repetition_words = {
            'certainty': ["真的", "确实", "的确", "果然", "当然", "必然", "肯定", "一定", "绝对"],
            'surprise': ["竟然", "居然", "没想到", "想不到", "意外", "奇怪"],
            'emphasis': ["就是", "正是", "恰好", "刚好", "偏偏", "专门", "特意"],
            'evaluation': ["不错", "还行", "挺好", "太好了", "糟糕", "可怕", "厉害"]
        }

        # Filler words and hesitation markers
        self.fillers = ["这个", "那个", "嗯", "呃", "额", "就是", "然后", "什么的", "之类的", "一类的", "等等"]

        # Regional variations and dialects
        self.regional_variants = {
            'northern': ["咋", "啥", "整", "弄", "搞", "得劲", "老", "贼"],
            'southern': ["蛮", "怪", "几", "撒", "搞", "弄", "整", "噶"],
            'internet_slang': ["绝了", "牛逼", "666", "厉害了", "棒棒哒", "么么哒"]
        }

    def _segment_with_pos_detailed(self, text: str) -> List[Tuple[str, str]]:
        """Enhanced POS tagging with detailed categories"""
        try:
            segments = list(pseg.cut(text))
            return segments
        except:
            return [(char, 'unknown') for char in text if char.strip()]

    def _add_discourse_particles_strategic(self, text: str) -> str:
        """Strategic placement of discourse particles"""
        if random.random() > 0.6:
            return text

        # Sentence end particles
        if random.random() < 0.4 and not re.search(r'[。！？]$', text):
            particle = random.choice(self.discourse_particles['sentence_end'])
            if text.endswith('。'):
                text = text[:-1] + particle + '。'
            elif text.endswith(('！', '？')):
                text = text[:-1] + particle + text[-1]
            else:
                text = text + particle

        # Sentence start particles
        if random.random() < 0.2 and len(text) > 5:
            particle = random.choice(self.discourse_particles['sentence_start'])
            text = particle + '，' + text

        # Mid-sentence particles
        if random.random() < 0.3 and len(text) > 15:
            particle = random.choice(self.discourse_particles['mid_sentence'])
            words = text.split('，')
            if len(words) > 1:
                insert_pos = random.randint(1, len(words) - 1)
                words[insert_pos] = particle + words[insert_pos]
                text = '，'.join(words)

        return text

    def _add_intensifiers_contextual(self, text: str) -> str:
        """Context-aware intensifier insertion"""
        segments = self._segment_with_pos_detailed(text)
        result = []

        i = 0
        while i < len(segments):
            word, pos = segments[i]

            # Before adjectives
            if pos in ['a', 'ad'] and random.random() < 0.4:
                # Check if already has intensifier
                if i > 0 and segments[i - 1][0] not in [item for sublist in self.intensifiers.values() for item in
                                                        sublist]:
                    intensifier_type = random.choice(list(self.intensifiers.keys()))
                    intensifier = random.choice(self.intensifiers[intensifier_type])
                    result.append(intensifier)

            # Before verbs for emphasis
            elif pos in ['v', 'vn'] and random.random() < 0.2:
                if word in ['是', '有', '会', '能', '要', '想', '喜欢', '讨厌']:
                    emphasis = random.choice(self.intensifiers['emphasis'])
                    result.append(emphasis)

            # Before negative constructions
            elif word in ['不', '没', '无'] and random.random() < 0.3:
                neg_intensifier = random.choice(self.intensifiers['negative'])
                result.append(neg_intensifier)
                continue  # Skip the original negation word

            result.append(word)
            i += 1

        return ''.join(result)

    def _add_connectors_logical(self, text: str) -> str:
        """Add logical connectors between clauses"""
        if len(text) < 20 or random.random() > 0.4:
            return text

        # Split by punctuation
        clauses = re.split(r'([，。！？；：])', text)
        clauses = [c for c in clauses if c.strip()]

        if len(clauses) < 3:  # Need at least text-punct-text
            return text

        # Choose connector type based on content analysis
        connector_type = 'temporal'  # default
        if any(word in text for word in ['因为', '由于', '因此', '所以']):
            connector_type = 'causal'
        elif any(word in text for word in ['但是', '不过', '然而']):
            connector_type = 'adversative'
        elif any(word in text for word in ['而且', '还有', '另外']):
            connector_type = 'additive'

        connector = random.choice(self.connectors[connector_type])

        # Find suitable insertion point
        for i in range(2, len(clauses), 2):  # Skip punctuation indices
            if clauses[i - 1] in ['，', '；'] and random.random() < 0.7:
                clauses[i] = connector + clauses[i]
                break

        return ''.join(clauses)

    def _add_modal_expressions(self, text: str) -> str:
        """Add modal particles and expressions"""
        if random.random() > 0.3:
            return text

        # Convert statements to questions
        if '吗' not in text and random.random() < 0.3:
            if text.endswith('。'):
                uncertainty = random.choice(self.modal_particles['uncertainty'])
                text = text[:-1] + uncertainty + '？'

        # Add confirmation particles
        elif random.random() < 0.2:
            confirmation = random.choice(self.modal_particles['confirmation'])
            text = text + confirmation

        # Add surprise expressions
        elif any(word in text for word in ['没想到', '意外', '竟然']) and random.random() < 0.4:
            surprise = random.choice(self.modal_particles['surprise'])
            text = surprise + '，' + text

        return text

    def _add_repetition_emphasis(self, text: str) -> str:
        """Add emphatic repetition and evaluation"""
        if random.random() > 0.25:
            return text

        # Add certainty expressions
        if random.random() < 0.4:
            certainty = random.choice(self.repetition_words['certainty'])
            if random.random() < 0.6:
                text = certainty + '，' + text
            else:
                # Insert in middle
                mid = len(text) // 2
                text = text[:mid] + certainty + '，' + text[mid:]

        # Add evaluation words
        elif random.random() < 0.3:
            evaluation = random.choice(self.repetition_words['evaluation'])
            text = text + '，' + evaluation

        return text

    def _add_fillers_hesitation(self, text: str) -> str:
        """Add filler words and hesitation markers"""
        if len(text) < 10 or random.random() > 0.2:
            return text

        words = text.split()
        if len(words) < 3:
            return text

        # Insert filler in middle
        filler = random.choice(self.fillers)
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, filler)

        return ''.join(words)

    def _add_regional_variants(self, text: str) -> str:
        """Add regional dialect variations"""
        if random.random() > 0.15:
            return text

        # Replace standard words with regional variants
        replacements = {
            '什么': ['啥', '撒', '么'],
            '怎么': ['咋', '咋样', '怎样'],
            '这样': ['这么', '酱紫', '这般'],
            '那样': ['那么', '那般'],
            '很好': ['很棒', '蛮好', '怪好的', '不错'],
            '厉害': ['牛逼', '666', '绝了', '太强了']
        }

        for standard, variants in replacements.items():
            if standard in text and random.random() < 0.5:
                variant = random.choice(variants)
                text = text.replace(standard, variant, 1)
                break

        return text

    def augment(self, text: str) -> str:
        """Apply comprehensive structural perturbation"""
        if random.random() > self.perturbation_rate:
            return text

        augmented = text.strip()

        # Apply different strategies with varying probabilities
        strategies = [
            (self._add_discourse_particles_strategic, 0.6),
            (self._add_intensifiers_contextual, 0.5),
            (self._add_connectors_logical, 0.4),
            (self._add_modal_expressions, 0.3),
            (self._add_repetition_emphasis, 0.3),
            (self._add_fillers_hesitation, 0.2),
            (self._add_regional_variants, 0.15)
        ]

        # Apply 1-4 strategies
        num_strategies = random.randint(1, min(4, len(strategies)))
        selected_strategies = random.sample(strategies, num_strategies)

        for strategy_func, prob in selected_strategies:
            if random.random() < prob:
                try:
                    augmented = strategy_func(augmented)
                except Exception as e:
                    logger.warning(f"Strategy {strategy_func.__name__} failed: {e}")
                    continue

        return augmented.strip() if augmented.strip() else text


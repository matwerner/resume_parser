from typing import List, Dict, Any
from collections import Counter

import sys
import numpy as np

from resume_parser.segmenter import text_features
from resume_parser.segmenter.dataset import TextElement


class LineFeatureExtractor(object):


    def __init__(self,
                 section_names_map: Dict[str,List[str]]={},
                 stop_words: List[str]=[],
                 use_vocabulary: bool=True,
                 use_text: bool=True,
                 use_visual: bool=True,
                 use_spatial: bool=True
                 ) -> None:
        self.stop_words = stop_words # Must be before section names
        self.section_names_map = section_names_map
        self.use_vocabulary = use_vocabulary
        self.use_text = use_text
        self.use_visual = use_visual
        self.use_spatial = use_spatial

        def normalize(texts: List[str]) -> List[str]:
            return {text_features.normalize_section_name(
                    text, self.stop_words) for text in texts}
        self.pd_names = normalize(self.section_names_map['Personal_Data'])
        self.we_names = normalize(self.section_names_map['Work_Experience'])
        self.edu_names = normalize(self.section_names_map['Education'])
        self.obj_names = normalize(self.section_names_map['Objective'])
        self.about_names = normalize(self.section_names_map['About'])
        self.other_names = normalize(self.section_names_map['Other'])


    def _extract_line_features_vocabulary(self,
                                          lines: List[TextElement],
                                          index: int,
                                          global_features: dict,
                                         ) -> Dict[str, float]:
        text = lines[index].text
        normalized_text = text_features.normalize_section_name(
            text, self.stop_words)
        vocabulary_features = [
            text_features.perc_section_name(normalized_text, self.pd_names),
            text_features.perc_section_name(normalized_text, self.we_names),
            text_features.perc_section_name(normalized_text, self.edu_names),
            text_features.perc_section_name(normalized_text, self.obj_names),
            text_features.perc_section_name(normalized_text, self.about_names),
            text_features.perc_section_name(normalized_text, self.other_names),
        ]
        return vocabulary_features


    def _get_vector_length_vocabulary(self):
        return 6 if self.use_vocabulary else 0


    def _extract_line_features_text(self,
                                    lines: List[TextElement],
                                    index: int,
                                    global_features: Dict[str, Any],
                                    ) -> dict:

        text = lines[index].text
        textual_features = [
            text_features.perc_isupper(text),
            text_features.perc_istitle(text),
            text_features.perc_islower(text),
            text_features.has_year(text),
            text_features.has_punct(text),
            text_features.has_punct_end(text),
            text_features.has_colon(text),
            text_features.has_colon_end(text),
            text_features.has_number(text),
            text_features.get_num_tokens(text, self.stop_words) <= 2,
        ]
        return textual_features


    def _get_vector_length_text(self):
        return 10 if self.use_text else 0


    def _extract_line_features_visual(self,
                                      lines: List[TextElement],
                                      index: int,
                                      global_features: Dict,
                                     ) -> Dict[str, Any]:
        line = lines[index]

        # Global properties
        default_fontsize = global_features['default_fontsize']
        min_fontsize = global_features['min_fontsize']
        max_fontsize = global_features['max_fontsize']
        default_rgb = global_features['default_rgb']

        rgb = line.rgb
        is_color_default = rgb == default_rgb

        fontsize = line.fontsize
        is_fontsize_default = fontsize == default_fontsize
        is_fontsize_larger = fontsize > default_fontsize
        is_fontsize_smaller = fontsize < default_fontsize

        def is_same_style(a: TextElement, b: TextElement) -> bool:
            if a is None or b is None:
                return True
            return a.bold == b.bold \
                and a.italic == b.italic \
                and a.rgb == b.rgb \
                and a.fontsize == b.fontsize \
                and a.text.isupper() == b.text.isupper() \
                and a.text.istitle() == b.text.istitle()

        prev_line = lines[index-1] if index > 0 else None
        is_prev_same = is_same_style(line, prev_line)

        next_line = lines[index+1] if index < len(lines) - 1 else None
        is_next_same = is_same_style(line, next_line)

        visual_features = [
            int(line.bold),
            int(line.italic),
            int(is_color_default),
            int(is_fontsize_default),
            int(is_fontsize_larger),
            int(is_fontsize_smaller),
            int(is_prev_same),
            int(is_next_same)
        ]

        return visual_features


    def _get_vector_length_visual(self):
        return 8 if self.use_visual else 0


    def _extract_line_features_spatial(self,
                                       lines: List[TextElement],
                                       index: int,
                                       global_features: Dict[str, Any],
                                      ) -> Dict[str, Any]:
        line = lines[index]

        # Global properties
        default_y_spacing = global_features['default_y_spacing']
        min_y_spacing = global_features['min_y_spacing']
        avg_y_spacing = global_features['avg_y_spacing']
        std_y_spacing = global_features['std_y_spacing']
        max_y_spacing = global_features['max_y_spacing']
        tab_width = global_features['tab_width']

        def get_features_wrt_default_spacing(y_spacing):
            greater_y_spacing = y_spacing > default_y_spacing
            lesser_y_spacing = y_spacing < default_y_spacing
            equal_y_spacing = not (greater_y_spacing or lesser_y_spacing)
            return [
                int(greater_y_spacing),
                int(equal_y_spacing),
                int(lesser_y_spacing)
            ]

        def get_features_wrt_average_spacing(y_spacing):
            greater_y_spacing = y_spacing > avg_y_spacing + std_y_spacing
            lesser_y_spacing = y_spacing < avg_y_spacing - std_y_spacing
            equal_y_spacing = not (greater_y_spacing or lesser_y_spacing)
            return [
                int(greater_y_spacing),
                int(equal_y_spacing),
                int(lesser_y_spacing)
            ]

        def get_features(prev_line: TextElement, line: TextElement):
            if prev_line is None or line is None:
                return 9 * [0]

            prev_line = lines[index-1]
            same_page = line.num_page == prev_line.num_page
            if same_page:
                y_spacing = line.y_coord - line.height - prev_line.y_coord
            else:
                y_spacing = default_y_spacing
            default_features = get_features_wrt_default_spacing(y_spacing)
            average_features = get_features_wrt_average_spacing(y_spacing)

            indent_minus = line.x_coord + tab_width < prev_line.x_coord
            indent_plus = line.x_coord - tab_width > prev_line.x_coord
            return [int(same_page), int(indent_minus), int(indent_plus)] \
                + default_features + average_features

        prev_line = lines[index-1] if index > 0 else None
        prev_features = get_features(prev_line, line)

        next_line = lines[index+1] if index < len(lines) - 1 else None
        next_features = get_features(line, next_line)

        return prev_features + next_features


    def _get_vector_length_spatial(self):
        return 18 if self.use_spatial else 0


    def _get_feature_vector(self,
                            lines: List[TextElement],
                            index: int,
                            global_features: dict,
                           ) -> Dict[str, Any]:
        extractor_fns = []
        if self.use_vocabulary: extractor_fns.append(self._extract_line_features_vocabulary)
        if self.use_text: extractor_fns.append(self._extract_line_features_text)
        if self.use_visual: extractor_fns.append(self._extract_line_features_visual)
        if self.use_spatial: extractor_fns.append(self._extract_line_features_spatial)

        features = []
        for extractor_fn in extractor_fns:
            features += extractor_fn(lines, index, global_features)
        return features


    def get_feature_vector_length(self):
        return 3 * (
            self._get_vector_length_vocabulary()
            + self._get_vector_length_text()
            + self._get_vector_length_spatial()
            + self._get_vector_length_visual()
        )


    def _compute_global_features(self, lines: List[TextElement]) -> dict:
        fontsizes, rgbs, y_spacings, char_widths = [], [], [], []
        left_margin = sys.maxsize
        for i, line in enumerate(lines):
            prev_line = lines[i-1] if i > 0 else None
            if prev_line and prev_line.num_page == line.num_page:
                delta = line.y_coord - line.height - prev_line.y_coord
                delta = round(delta, 0)
                y_spacings.append(delta)
            fontsizes.append(line.fontsize)
            rgbs.append(line.rgb)
            char_widths.append(int(line.width / len(line.text)))
            left_margin = min(left_margin, line.x_coord)
        used_fontsizes = sorted(list(set(fontsizes)))

        def get_most_common(array, default_value=None):
            if len(array) == 0:
                return default_value
            return Counter(array).most_common(1)[0][0]

        resume_features = {
            # Fontsize
            'used_fontsizes': used_fontsizes,
            'default_fontsize': get_most_common(fontsizes),
            'min_fontsize': np.min(used_fontsizes),
            'avg_fontsize': np.mean(fontsizes),
            'std_fontsize': np.std(fontsizes),
            'max_fontsize': np.max(used_fontsizes),

            # Color
            'default_rgb': get_most_common(rgbs),

            # Y-spacing
            'default_y_spacing': get_most_common(y_spacings, 0),
            'min_y_spacing': np.min(y_spacings),
            'avg_y_spacing': np.mean(y_spacings),
            'std_y_spacing': np.std(y_spacings),
            'max_y_spacing': np.max(y_spacings),

            # X-spacing
            'tab_width': 1 * get_most_common(char_widths), # tab = 2~4 spaces
            'left_margin': left_margin,
        }
        return resume_features


    def get_feature_vectors(self, words: List[TextElement]) -> List[dict]:
        lines = self._to_lines(words)
        global_features = self._compute_global_features(lines)
        line_vectors = [self._get_feature_vector(lines, i, global_features)
                        for i in range(len(lines))]
        ex_line_vectors = []
        for i, line_vector in enumerate(line_vectors):
            prev_line_vector = line_vectors[i-1] if i > 0 else ([0] * len(line_vector))
            next_line_vector = line_vectors[i+1] if i < len(lines) - 1 else ([0] * len(line_vector))
            ex_line_vector = prev_line_vector + line_vector + next_line_vector
            ex_line_vectors.append(ex_line_vector)
        # Add bias
        return [ex_line_vectors[word.line_idx] for word in words]


    def _to_lines(self, words: List[TextElement]) -> List[TextElement]:
        # Group words per line
        line_words_map = {}
        for word in words:
            line_idx = word.line_idx
            if line_idx not in line_words_map:
                line_words_map[line_idx] = []
            line_words_map[line_idx].append(word)

        # Words -> Line
        lines = [self._merge_words(line_words)
                 for line_words in line_words_map.values()]

        lines = sorted(lines, key=lambda l: l.line_idx)
        return lines


    def _merge_words(self, words: List[TextElement]) -> TextElement:
        # Text
        text = ''.join(w.text + (w.tail if i+1 != len(words) else '')
                    for i, w in enumerate(words))
        tail = words[-1].tail
        start = min(w.start for w in words)
        end = max(w.end for w in words)
        # Visual - Get most frequent attribute
        def get_most_common(array: List):
            return Counter(array).most_common(1)[0][0]
        fontsize = get_most_common([w.fontsize for w in words])
        bold = get_most_common([w.bold for w in words])
        italic = get_most_common([w.italic for w in words])
        rgb = get_most_common([w.rgb for w in words])
        # Bounding box - Top-left orientation
        x_coord = min(w.x_coord for w in words)
        y_coord = max(w.y_coord for w in words)
        width = max(w.x_coord + w.width - x_coord for w in words)
        height = max(w.height + y_coord - w.y_coord for w in words)
        num_page = min(w.num_page for w in words)
        line_indices = list(set(w.line_idx for w in words))
        assert len(line_indices) == 1, \
            f"Elements merged must be from the same line. Text: {text}"
        line_idx = line_indices[0]

        return TextElement(
            text = text,
            start = start,
            end = end,
            fontsize = fontsize,
            bold = bold,
            italic = italic,
            rgb = rgb,
            x_coord = x_coord,
            y_coord = y_coord,
            width = width,
            height = height,
            num_page = num_page,
            word_idx = -1,
            line_idx = line_idx,
            tail = tail,
            label = None
        )

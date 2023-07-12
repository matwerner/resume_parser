from collections import ChainMap, Counter
from typing import List, Dict

import re
import sys
import nltk
import numpy as np
import pandas as pd
import unidecode

from joblib import Parallel, delayed
from sklearn import feature_extraction, feature_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _transform_one

from resume_parser.segmenter import text_features
from resume_parser.segmenter.dataset import TextElement


class DictFeatureUnion(FeatureUnion):

    def __init__(self,
                 transformer_list,
                 *,
                 n_jobs=None,
                 transformer_weights=None,
                 verbose=False):
        super().__init__(transformer_list, n_jobs=n_jobs,
                         transformer_weights=transformer_weights,
                         verbose=verbose)

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix of
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return X.shape[0] * [[]]
        return [[dict(ChainMap(*feature_dicts)) for feature_dicts in zip(*x_is)]
                 for x_is in zip(*Xs)]


class LineFeatureExtractor(TransformerMixin, BaseEstimator):
    FONTSIZE_THRESHOLD = 0.01
    X_COORD_THRESHOLD = 1


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


    def _is_same_style_1(self, a: TextElement, b: TextElement) -> bool:
        return a.bold == b.bold \
           and a.italic == b.italic \
           and a.rgb == b.rgb \
           and a.fontsize == b.fontsize \
           and a.text.isupper() == b.text.isupper() \
           and a.text.istitle() == b.text.istitle()

    def _is_same_style_2(self, a: TextElement, b: TextElement) -> bool:
        return a.bold == b.bold \
           and a.italic == b.italic \
           and a.rgb == b.rgb \
           and a.fontsize == b.fontsize

    def _is_same_style_3(self, a: TextElement, b: TextElement) -> bool:
        return a.bold == b.bold \
           and a.italic == b.italic \
           and a.rgb == b.rgb \

    def _minmax_scale(self, x, x_min, x_max):
        std_x = (x - x_min) / (x_max - x_min) # [0, 1]
        scaled_x = 2 * std_x - 1 # [-1, 1]
        return scaled_x


    def _standard_scale(self, x, x_avg, x_std):
        return (x - x_avg) / x_std


    def _extract_line_features_vocabulary(self,
                                          lines: List[TextElement],
                                          index: int,
                                          global_features: dict,
                                          is_before: bool=False,
                                          is_after: bool=False
                                         ) -> dict:

        text = lines[index].text
        normalized_text = text_features.normalize_section_name(text, self.stop_words)
        vocabulary_features = {
            'vocabulary.perc_personal_data()': text_features.perc_section_name(normalized_text, self.pd_names_),
            'vocabulary.perc_work_experience()': text_features.perc_section_name(normalized_text, self.we_names_),
            'vocabulary.perc_education()': text_features.perc_section_name(normalized_text, self.edu_names_),
            'vocabulary.perc_objective()': text_features.perc_section_name(normalized_text, self.obj_names_),
            'vocabulary.perc_about()': text_features.perc_section_name(normalized_text, self.about_names_),
            'vocabulary.perc_other()': text_features.perc_section_name(normalized_text, self.other_names_),
        }
        return vocabulary_features


    def _extract_line_features_text(self,
                                    lines: List[TextElement],
                                    index: int,
                                    global_features: dict,
                                    is_before: bool=False,
                                    is_after: bool=False
                                   ) -> dict:

        text = lines[index].text
        textual_features = {
            # 'text.isupper()': text.isupper(),
            # 'text.istitle()': text.istitle(),
            'text.perc_isupper()': text_features.perc_isupper(text),
            'text.perc_istitle()': text_features.perc_istitle(text),
            'text.perc_islower()': text_features.perc_islower(text),
            'text.has_year()': text_features.has_year(text),
            'text.has_punct()': text_features.has_punct(text),
            'text.has_punct_end()': text_features.has_punct_end(text),
            'text.has_colon()': text_features.has_colon(text),
            'text.has_colon_end()': text_features.has_colon_end(text),
            'text.has_number()': text_features.has_number(text),
            'text.has_less_3_tokens()': text_features.get_num_tokens(text, self.stop_words) <= 2,
        }
        return textual_features


    def _extract_line_features_visual(self,
                                      lines: List[TextElement],
                                      index: int,
                                      global_features: dict,
                                      is_before: bool=False,
                                      is_after: bool=False
                                      ) -> dict:

        line = lines[index]

        # Global properties
        default_fontsize = global_features['default_fontsize']
        # min_fontsize = global_features['min_fontsize']
        # max_fontsize = global_features['max_fontsize']
        default_rgb = global_features['default_rgb']

        rgb = line.rgb
        fontsize = line.fontsize

        visual_features = {
            'visual.bold': line.bold,
            'visual.italic': line.italic,
            'visual.fontsize': fontsize,
            'visual.rgb = default': rgb == default_rgb,
            'visual.fontsize = default': fontsize == default_fontsize,
            'visual.fontsize > default': fontsize > default_fontsize,
            'visual.fontsize < default': fontsize < default_fontsize,
        }

        if index > 0 and not is_after:
            prev_line = lines[index-1]
            visual_features['-1:visual.same_style_1'] = self._is_same_style_1(line, prev_line)
            # visual_features['-1:visual.same_style_2'] = self._is_same_style_2(line, prev_line)
            # visual_features['-1:visual.same_style_3'] = self._is_same_style_3(line, prev_line)
        if index < len(lines) - 1 and not is_before:
            next_line = lines[index+1]
            visual_features['+1:visual.same_style_1'] = self._is_same_style_1(line, next_line)
            # visual_features['+1:visual.same_style_2'] = self._is_same_style_2(line, next_line)
            # visual_features['+1:visual.same_style_3'] = self._is_same_style_3(line, next_line)
        return visual_features


    def _extract_line_features_spatial(self,
                                       lines: List[TextElement],
                                       index: int,
                                       global_features: dict,
                                       is_before: bool=False,
                                       is_after: bool=False
                                      ) -> Dict:

        line = lines[index]

        # Global properties
        default_y_spacing = global_features['default_y_spacing']
        min_y_spacing = global_features['min_y_spacing']
        avg_y_spacing = global_features['avg_y_spacing']
        std_y_spacing = global_features['std_y_spacing']
        max_y_spacing = global_features['max_y_spacing']
        tab_width = global_features['tab_width']
        # Not a good feature for dealing with 2-Column resumes
        left_margin = global_features['left_margin']

        def get_features_wrt_default_spacing(y_spacing):
            greater_y_spacing = y_spacing > default_y_spacing
            lesser_y_spacing = y_spacing < default_y_spacing
            equal_y_spacing = not (greater_y_spacing or lesser_y_spacing)
            return {
                'spatial.y_spacing > default': greater_y_spacing,
                'spatial.y_spacing = default': equal_y_spacing,
                'spatial.y_spacing < default': lesser_y_spacing
            }

        def get_features_wrt_average_spacing(y_spacing):
            greater_y_spacing = y_spacing > avg_y_spacing + std_y_spacing
            lesser_y_spacing = y_spacing < avg_y_spacing - std_y_spacing
            equal_y_spacing = not (greater_y_spacing or lesser_y_spacing)
            return {
                'spatial.y_spacing > avg + 1 std': greater_y_spacing,
                'spatial.y_spacing = avg ± 1 std': equal_y_spacing,
                'spatial.y_spacing < avg + 1 std': lesser_y_spacing
            }

        def get_features(prev_line: TextElement, line: TextElement):
            if prev_line is None or line is None:
                return {}

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

            features = {
                'same_page': same_page,
                'spatial.indent-1': indent_minus,
                'spatial.indent+1': indent_plus
            }
            features.update(default_features)
            features.update(average_features)
            return features

        prev_line = lines[index-1] if index > 0 and not is_after else None
        prev_features = get_features(prev_line, line)

        next_line = lines[index+1] if index < len(lines)-1 and not is_before else None
        next_features = get_features(line, next_line)

        spatial_features = {}
        spatial_features.update({f'-1:{key}': value
                                 for key, value in prev_features.items()})
        spatial_features.update({f'+1:{key}': value
                                 for key, value in next_features.items()})
        return spatial_features


    def _get_feature_vector(self,
                            lines: List[TextElement],
                            index: int,
                            global_features: dict,
                            is_before: bool=False,
                            is_after: bool=False
                           ) -> dict:
        extractor_fns = []
        if self.use_vocabulary: extractor_fns.append(self._extract_line_features_vocabulary)
        if self.use_text: extractor_fns.append(self._extract_line_features_text)
        if self.use_visual: extractor_fns.append(self._extract_line_features_visual)
        if self.use_spatial: extractor_fns.append(self._extract_line_features_spatial)

        features = {'bias': 1.0}
        for extractor_fn in extractor_fns:
            features.update(extractor_fn(lines, index, global_features,
                is_before, is_after))
        return features


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


    def _get_feature_vectors(self, x: List[TextElement]) -> List[dict]:
        global_features = self._compute_global_features(x)
        return [self._get_feature_vector(x, i, global_features) for i in range(len(x))]


    def fit(self, X: List[List[TextElement]], y=None):
        def normalize(texts: List[str]) -> List[str]:
            return {text_features.normalize_section_name(text, self.stop_words)
                    for text in texts}
        self.pd_names_ = normalize(self.section_names_map['Personal_Data'])
        self.we_names_ = normalize(self.section_names_map['Work_Experience'])
        self.edu_names_ = normalize(self.section_names_map['Education'])
        self.obj_names_ = normalize(self.section_names_map['Objective'])
        self.about_names_ = normalize(self.section_names_map['About'])
        self.other_names_ = normalize(self.section_names_map['Other'])

        self.section_names_ = self.pd_names_
        self.section_names_ |= self.we_names_
        self.section_names_ |= self.edu_names_
        self.section_names_ |= self.obj_names_
        self.section_names_ |= self.about_names_
        self.section_names_ |= self.other_names_
        return self


    def transform(self, X: List[List[TextElement]]):
        return [self._get_feature_vectors(x) for x in X]


class MultiLineFeatureExtractor(LineFeatureExtractor):


    def __init__(self,
                 section_names_map: Dict[str, List[str]]={},
                 stop_words: List[str]=[],
                 use_vocabulary: bool=True,
                 use_text: bool=True,
                 use_visual: bool=True,
                 use_spatial: bool=True
                 ) -> None:
        super().__init__(section_names_map, stop_words, use_vocabulary,
                         use_text, use_visual, use_spatial)


    def _get_feature_vector(self,
                            lines: List[TextElement],
                            index: int,
                            global_features: dict
                           ) -> dict:
        features = super()._get_feature_vector(lines, index, global_features)
        if index > 0:
            features_before = super()._get_feature_vector(lines, index-1,
                global_features, is_before=True)
            features_before.pop('bias', None) # Duplicate feature
            features.update({'-1:'+key: value for key, value in features_before.items()})
        else:
            features['BOS'] = True

        if index < len(lines) - 1:
            features_after = super()._get_feature_vector(lines, index+1,
                global_features, is_after=True)
            features_after.pop('bias', None) # Duplicate feature
            features.update({'+1:'+key: value for key, value in features_after.items()})
        else:
            features['EOS'] = True

        return features


class TextPreprocessor(TransformerMixin, BaseEstimator):


    def __init__(self, is_stemm: bool=False, stop_words: List[str]=None) -> None:
        self.is_stemm = is_stemm
        self.stop_words = stop_words


    def fit(self, X: List[List[TextElement]], y: List[str]=None):
        return self


    def transform(self, X: List[List[TextElement]]):
        return [[self._clean_text(line.text) for line in x_i] for x_i in X]


    def _clean_text(self, text: str) -> str:
        ''' Preprocess a string.

        Parameters
        ----------
            text : str
                name of column containing text
            flg_stemm : bool
                whether stemming is to be applied
            lst_stopwords : list
                list of stopwords to remove
        Returns
        -------
            str
                cleaned text
        '''

        # convert to lowercase and strip
        text = str(text).lower().strip()

        ## clean (remove punctuations and characters)
        text = re.sub(r'[^\w\s]', ' ', unidecode.unidecode(text))
        text = re.sub(r'[0-9]', '0', text)

        ## Tokenize (convert from string to list)
        lst_text = text.split()

        ## remove Stopwords
        if self.stop_words is not None:
            lst_text = [word for word in lst_text
                        if word not in self.stop_words]

        ## Stemming (remove -ndo, -ou, ...)
        if self.is_stemm:
            ps = nltk.stem.RSLPStemmer()
            lst_text = [ps.stem(word) for word in lst_text]

        ## back to string from list
        text = ' '.join(lst_text)
        return text


class SelectTfidfFeatureExtractor(TransformerMixin, BaseEstimator):


    def __init__(self, max_features=-1, max_features_label=-1,
                 ngram_range=(1,1), debug: bool=False) -> None:
        self.max_features = max_features
        self.max_features_label = max_features_label
        self.ngram_range = ngram_range
        self.debug = debug


    def _to_level_line(self, X: List[List[str]], y: List[List[str]]=None):
        X = np.array([line for x_i in X for line in x_i])
        # y = np.array([label for y_i in y for label in y_i])
        y = np.array([re.sub(r'^[IOB]-', '', label) for y_i in y for label in y_i])
        return X, y


    def _to_level_section(self, X: List[List[str]], y: List[List[str]]=None):
        y = [[re.sub(r'^[IOB]-', '', label) for label in y_i] for y_i in y]
        records = [(i, line, label) for i, (X_i, y_i) in enumerate(zip(X, y))
                   for line, label in zip(X_i, y_i)]

        df = pd.DataFrame(records, columns=['index', 'line', 'label'])
        df = (df.groupby(['index', 'label'])['line']
                .agg(' '.join)
                .reset_index(level=0, drop=True)
                .reset_index(name='lines'))
        X = df['lines'].values
        y = df['label'].values
        return X, y


    def fit(self, X: List[List[str]], y: List[List[str]]=None):
        # feature_extraction.text.CountVectorizer
        # feature_extraction.text.TfidfVectorizer
        self.vectorizer_ = feature_extraction.text.TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            binary=True,
            min_df=100
        )

        # Document-level
        X, y = self._to_level_line(X, y)

        self.vectorizer_.fit(X)
        if y is None:
            return self

        X_tfidf = self.vectorizer_.transform(X)
        X_names = self.vectorizer_.get_feature_names()

        p_value_limit = 0.99
        features_df = pd.DataFrame()
        for label in np.unique(y):
            _, p = feature_selection.chi2(X_tfidf, y==label)
            label_features_dict = {"feature":X_names, "score":1-p, "y":label}
            label_features_df = pd.DataFrame(label_features_dict)
            label_features_df = label_features_df.sort_values(["y","score"], ascending=[True,False])
            label_features_df = label_features_df[label_features_df["score"] > p_value_limit]
            label_features_df = label_features_df.iloc[:self.max_features_label]

            features_df = pd.concat([features_df, label_features_df])
        features_df = features_df.sort_values(["y","score"],ascending=[True,False])
        X_names = features_df["feature"].unique().tolist()

        if self.debug:
            print("# Selected features: ", len(features_df))
            for label in np.unique(y):
                print("# {}:".format(label))
                print("  . selected features:", len(features_df[features_df["y"]==label]))
                print("  . top features:", ",".join(features_df[features_df["y"]==label]["feature"].values[:50]))
                print(" ")

        self.vectorizer_ = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
        self.vectorizer_.fit(X)
        return self


    def _transform_one(self, x_i: List[str]):
        Mc = self.vectorizer_.transform(x_i).tocoo()

        # To list of dicts
        feature_names = self.vectorizer_.get_feature_names()
        x_t = [dict() for _ in range(len(x_i))]
        for i, j, value in zip(Mc.row, Mc.col, Mc.data):
            x_t[i][feature_names[j]] = value
        return x_t


    def transform(self, X: List[List[str]]):
        return [self._transform_one(x_i) for x_i in X]

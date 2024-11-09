"""main.py

Defines the NBSpaceRestorer class"""

import operator
import time
from collections import Counter
from functools import lru_cache, reduce
from math import log10
from typing import List, Optional, Tuple, Union

from nltk.util import everygrams
import pandas as pd
import psutil
from fre import FeatureRestorationEvaluator
from sklearn.model_selection import ParameterGrid
from scipy.stats import norm
import numpy as np

from nb_space_restorer.nb_helper import (display_or_print, get_tqdm,
                                         load_pickle, save_pickle,
                                         try_clear_output)

tqdm_ = get_tqdm()

MAX_CACHE_SIZE = 1_000_000
L_DEFAULT = 20
LAMBDA_DEFAULT = 10.0
METRIC_TO_OPTIMIZE_DEFAULT = 'F-score'
MIN_OR_MAX_DEFAULT = 'max'

ERROR_MIN_OR_MAX = """\
min_or_max should be one of either "min" or "max"
"""

MESSAGE_FINISHED_LOADING = "Finished loading model."
MESSAGE_GRID_SEARCH_INCOMPLETE = """\
Grid search {grid_search_name} is incomplete. There are {num_untested} \
parameter combination(s) that have not been tested. To resume the grid \
search, call the run_grid_search method with the same reference and \
input texts you used when you added the grid search."""
MESSAGE_OPTIMAL_PARAMS = """\
Optimal hyperparameter values based on the results of the current grid
search are L={L} and lambda={lambda_}. Run set_optimal_params to set these
values for the current model.
"""
MESSAGE_SAVED = "Model saved to {}."
MESSAGE_L_SET = """\
The value of the hyperparameter L was set to: {L}."""
MESSAGE_LAMBDA_SET = """\
The value of the hyperparameter lambda was set to: {lambda_}."""
MESSAGE_FUNCTION_SET = """\
The value of the hyperparameter lambda was set to: {unknown_function}."""
MESSAGE_METRIC_TO_OPTIMIZE_SET = """\
The metric to optimize was set to: '{metric_to_optimize}'."""
MESSAGE_MIN_OR_MAX_SET = """\
The setting of whether to minimize or maximize the optimization
metric was set to: {min_or_max}."""
MESSAGE_SKIPPING_PARAMS = """\
Skipping parameter combination at index {i} because results \
are already in the grid search log."""
MESSAGE_TESTED_SO_FAR = """\
{completed}/{total} parameter combinations tested so far."""
MESSAGE_TRAINING_COMPLETE = "Training complete."


# ====================
class NBSpaceRestorer:

    # ====================
    def __init__(self,
                 train_texts: list,
                 uyir_letters: list[str],
                 mei_letters: list[str],
                 ignore_case: bool = True,
                 save_path: Optional[str] = None,
                 max_n_gram: int = 2,
                 unknown_function: str = 'exponential'):
        """Initialize and train an instance of the class.

        Args:
          train_texts (list):
            The list of 'gold standard' documents (running text with spaces)
            on which to train the model.
          uyir_letters (list):
            A list of letters that can only appear at the beginning of words
          mei_letters (list):
            A list of letters that can never appear at the beginning of words
          ignore_case (bool, optional):
            Whether or not to ignore case during training (so that e.g.
            'banana', 'Banana', and 'BANANA' are all counted as instances
            of 'banana'). Defaults to True.
          save_path (Optional[str], optional):
            The path to a pickle file to save the model to. Defaults to None.
          max_n_gram (int, optional):
            The maximum value of N to calculate all N-grams for. Defaults to 2.
          unknown_function (str, optional):
            The function to use to predict the probability of an unseen word.
            Can be 'exponential' or 'gaussian', defaults to 'exponential'
        """

        self.save_path = save_path
        self.L = L_DEFAULT
        self.lambda_ = LAMBDA_DEFAULT
        self.metric_to_optimize = METRIC_TO_OPTIMIZE_DEFAULT
        self.min_or_max = MIN_OR_MAX_DEFAULT
        self.running_grid_search = False
        self.freqs: Counter = Counter()
        self.ngram_freqs: Counter = Counter()
        self.max_n_gram: int = max_n_gram

        self.uyir_letters = uyir_letters.copy()
        self.mei_letters = mei_letters.copy()

        self.distribution = {}
        self.distribution_fn = None
        self.likely_len = 0
        word_lengths = []

        for text in train_texts:
            if ignore_case:
                text = text.lower()
            words = text.split()
            for word in words:
                word_lengths.append(len(word))

            grams = list(everygrams(words, max_len=max_n_gram))
            self.freqs.update(grams)
            self.ngram_freqs.update([len(gram) for gram in grams])

        self.distribution_fn = norm(np.mean(word_lengths), np.std(word_lengths))
        self.likely_len = np.mean(word_lengths)
        for length in range(1, 20):
            self.distribution[length] = self.distribution_fn.pdf(length)

        self.unknown_function = unknown_function

        self.grid_searches = {}
        self.get_pdists()
        print(MESSAGE_TRAINING_COMPLETE)
        self.save()

    # ====================
    def save(self):
        """If self.save_path is defined, save the model attributes to that
        path
        """

        if self.save_path is not None:
            save_pickle(self.__dict__, self.save_path)
            print(MESSAGE_SAVED.format(self.save_path))

    # ====================
    @classmethod
    def load(cls,
             load_path: str,
             read_only: bool = False) -> 'NBSpaceRestorer':
        """Load a previously saved instance of the class.

        Args:
          load_path (str):
            The path to the pickle file that contains the model
            attributes
          read_only (bool, optional):
            If set to True, the model will be loaded but changes made after
            loading will not be written back to the pickle file.

        Returns:
          NBSpaceRestorer:
            The loaded class instance
        """

        self = cls.__new__(cls)
        self.__dict__ = load_pickle(load_path)
        if read_only is True:
            self.save_path = None
        else:
            self.save_path = load_path
        print(MESSAGE_FINISHED_LOADING)
        self.save()

        return self

    # ====================
    def get_pdists(self):
        """Get N-gram probability distributions from N-gram
        frequencies"""

        self.PNdist = {gram: freq / self.ngram_freqs[len(gram)]
                       for gram, freq in self.freqs.items()}

    # ====================
    def splits(self, text: str) -> List[tuple]:
        """Split text into a list of candidate (word, remainder) pairs.

        Args:
          text (str):
            An unspaced input text

        Returns:
          List[tuple]:
            A list of candidate (word, remainder) pairs
        """
        stop_index = self.L
        for index, character in enumerate(text[1:self.L]):
            if character in self.uyir_letters:
                stop_index = index + 1
                break

        return [
            (text[:i + 1], text[i + 1:]) for i in range(min(len(text), stop_index))
        ]

    # ====================
    @staticmethod
    def product(lis_: List[float]) -> float:
        """Product of a list of numbers

        Args:
          lis_ (List[float]):
            A list of floats

        Returns:
          float:
            The product of the floats in the input list
        """

        return reduce(operator.mul, lis_, 1)

    # ====================
    def Pwords(self, words: List[str]) -> float:
        """Get Naive Bayes probability of a sequence of words.

        Args:
          words (List[str]):
            A list of words

        Returns:
          float:
            The NB probability
        """

        return self.product([self.Pw(w) for w in words])

    # ====================
    def Pw(self, word: str) -> float:
        """Get Naive Bayes probability of a single word

        Args:
          word (str):
            A single candidate word

        Returns:
          float:
            The NB probability
        """

        if (word,) in self.PNdist:
            return self.PNdist[(word,)]
        else:
            # For unknown words, assign lower probabilities for longer words
            if self.unknown_function == 'gaussian':
                return self.lambda_ / self.ngram_freqs[1] *\
                                    (self.distribution[len(word)] if len(word)
                                    in self.distribution else self.distribution_fn(len(word)))
            else:
                return self.lambda_ / (self.ngram_freqs[1] * 10 ** (abs(len(word) - self.likely_len)))

    # ====================
    # def cPw(self, word: str, prev: str) -> float:
    def cPw(self, words: tuple[str]) -> float:
        """Get the conditional probability of a word given the previous word.

        Args:
          words (tuple[str]):
            The candidate word preceded by its previous words

        Returns:
            float: The Naive Bayes probability
        """
        if words[-1][0] in self.mei_letters:
            return 1e-100

        if len(words) == 1:
            return self.Pw(words[0])

        if words in self.freqs and words[:-1] in self.freqs:
            return self.freqs[words] / self.freqs[words[:-1]]
        else:
            return self.cPw(words[1:])

        # try:
        #     return self.PNdist[words] / self.PNdist[words[:-1]]
        # except KeyError:
        #     return self.lambda_ * self.cPw(words[1:])
        #     # return self.cPw(words[1:])

    # ====================
    def combine(self,
                Pfirst: float,
                first: str,
                rem_: Tuple[float, list]) -> Tuple[float, list]:
        """Combine the probability of a word, the word, and list of remaining
        words together with their probability to return the combined
        probability and combined list of words.

        Args:
          Pfirst (float):
            Probability of the first word
          first (str):
            The first word
          rem_ (Tuple[float, list]):
            The probability of the remaining words, and the list of words

        Returns:
          Tuple[float, list]:
            The combined probability and combined list of words
        """

        Prem, rem = rem_
        return Pfirst + Prem, [first] + rem

    # ====================
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def restore_chunk(self, text_: str, prev: tuple[str] = None) -> Tuple[float, list]:
        """Restore spaces to a short string of input characters

        Will result in RecursionError if length of text_ is more than
        around 100.

        Args:
          text_ (str):
            The text to restore spaces to.
          prev (tuple[str], optional):
            The previous words. Defaults to ('<S>',) repeated max_n_gram times.

        Returns:
          Tuple[float, list]:
            The probability of the most likely split, and the list of
            words
        """

        if prev is None:
            prev = ('<S>',) * self.max_n_gram

        if not text_:
            return 0.0, []

        candidates = [self.combine(log10(self.cPw(prev[1:] + (first,))),
                                   first,
                                   self.restore_chunk(rem, prev[1:] + (first,)))
                      for first, rem in self.splits(text_)]
        return max(candidates)

    # ====================
    def restore_doc(self,
                    text: str,
                    show_chunks: bool = False) -> str:
        """Restore spaces to a string of input characters of arbitrary
        length.

        For strings over around 100 characters in length, break them
        into chunks for segmentation and then put the words back together
        to avoid recursion limit errors.

        Args:
          text (str):
            The text to restore spaces to
          show_chunks (bool, optional):
            Whether to print out information about each chunk.
            Defaults to False.

        Returns:
          str:
            The document with spaces restored
        """

        chunk_len_chars = 80  # Low enough to avoid recursion errors
        all_words = []
        prefix = ''
        chunk_counter = 1
        # Iterate over chunks of the input string
        for offset in range(0, len(text), chunk_len_chars):
            # Prefix with the last five 'words' from the previous segmentation
            text_to_segment = prefix + text[offset:offset + chunk_len_chars]
            chunk_segmented = self.restore_chunk(text_to_segment)[1].copy()
            # Words may have been cut off at the end, so put the last five
            # words back into the segmenter next time round and discard them
            # this time
            prefix = ''.join(chunk_segmented[-5:])
            all_words.extend(chunk_segmented[:-5])
            if show_chunks:
                print(f'Chunk {chunk_counter}')
                print(f'Text segmented: {text_to_segment}')
                print(f'Result of segmentation: {chunk_segmented}')
                print(f'Words added to list this time: {chunk_segmented[:-5]}')
                print(f'Prefix for next chunk: {prefix}')
                print(f'Words added to list so far: {all_words}')
                print('-' * 100)
                chunk_counter += 1
        # Add any text remaining in 'prefix'
        all_words.extend(self.restore_chunk(prefix)[1])
        joined = ' '.join(all_words).strip()
        return joined

    # ====================
    def restore(self,
                texts: Union[str, List[str]],
                L: Optional[int] = None,
                lambda_: Optional[int] = None) -> Union[str, List[str]]:
        """Restore spaces to either a single string, or a list of
        strings.

        Args:
          texts (Union[str, List[str]]):
            Either a single string of input characters not containing spaces
            (e.g. 'thisisasentence') or a list of such strings
          L (Optional[int], optional):
            The value of the hyperparameter L to set before restoring
          lambda_ (Optional[float], optional):
            The value of the hyperparameter lambda_ to set before restoring

        Returns:
          Union[str, List[str]]:
            The string or list of strings with spaces restored
        """

        self.set_L(L)
        self.set_lambda(lambda_)
        if isinstance(texts, str):
            return self.restore_doc(texts)
        if isinstance(texts, list):
            restored = []
            texts_ = tqdm_(texts)
            for text in texts_:
                restored_ = self.restore_doc(text)
                restored.append(restored_)
                cache_size = self.restore_chunk.cache_info().currsize
                texts_.set_postfix({
                    'ram_usage': f"{psutil.virtual_memory().percent}%",
                    'cache_size': f"{cache_size:,}"
                })
            return restored

    # ====================
    def set_L(self, L: Union[int, None]):

        if L is None:
            return
        self.L = int(L)
        if self.running_grid_search is False:
            print(MESSAGE_L_SET.format(L=L))
            self.save()

    # ====================
    def set_lambda(self, lambda_: Union[float, None]):

        if lambda_ is None:
            return
        self.lambda_ = float(lambda_)
        if self.running_grid_search is False:
            print(MESSAGE_LAMBDA_SET.format(lambda_=lambda_))
            self.save()

    # ====================
    def set_unknown_function(self, unknown_function: Union[str, None]):

        if unknown_function is None:
            return
        self.unknown_function = str(unknown_function)
        if self.running_grid_search is False:
            print(MESSAGE_FUNCTION_SET.format(unknown_function=unknown_function))
            self.save()

    # ====================
    def set_metric_to_optimize(self, metric_to_optimize: str):

        if metric_to_optimize is None:
            return
        self.metric_to_optimize = metric_to_optimize
        print(MESSAGE_METRIC_TO_OPTIMIZE_SET.format(
            metric_to_optimize=metric_to_optimize
        ))
        self.save()

    # ====================
    def set_min_or_max(self, min_or_max: str):

        if min_or_max is None:
            return
        min_or_max = min_or_max.lower()
        if min_or_max not in ['min', 'max']:
            raise ValueError(ERROR_MIN_OR_MAX)
        self.min_or_max = min_or_max
        print(MESSAGE_MIN_OR_MAX_SET.format(min_or_max=min_or_max))
        self.save()

    # === GRID SEARCH ===

    # ====================
    def add_grid_search(self,
                        grid_search_name: str,
                        L: List[int],
                        lambda_: List[float],
                        ref: List[str],
                        input: List[str]):
        """Add and start running a grid search to find optimal hyperparameters
        for the model.

        Args:
          grid_search_name (str):
            A name for the grid search (e.g. 'grid_search_1')
          L (List[int]):
            A list of values for the hyperparameter L.
            (E.g. [18, 20, 22])
          lambda_ (List[float]):
            A list of values for the hyperparameter lambda.
            (E.g. [8.0, 10.0, 12.0])
          ref (List[str]):
            A list of reference documents to use in the grid search
          input (List[str]):
            A list of input documents to use in the grid search. Should be the
            same as the reference documents, but with spaces removed.
        """

        self.grid_searches[grid_search_name] = {}
        self.current_grid_search_name = grid_search_name
        self.current_grid_search()['param_values'] = {
            'L': L,
            'lambda': lambda_
        }
        param_combos = list(ParameterGrid({
            'L': L,
            'lambda': lambda_
        }))
        self.current_grid_search()['param_combos'] = \
            {i: pc for i, pc in enumerate(param_combos)}
        self.current_grid_search()['results'] = \
            {i: None for i in range(len(param_combos))}
        self.save()
        self.run_grid_search(ref, input)

    # ====================
    def load_grid_search(self, grid_search_name: str):

        self.grid_search_name = grid_search_name
        completed, total = self.param_combos_completed()
        if total > completed:
            print(MESSAGE_GRID_SEARCH_INCOMPLETE.format(
                grid_search_name=grid_search_name,
                num_untested=total - completed
            ))

    # ====================
    def current_grid_search(self):

        return self.grid_searches[self.current_grid_search_name]

    # ====================
    def run_grid_search(self, ref: List[str], input: List[str]):

        param_combos = self.current_grid_search()['param_combos']
        L_start = self.L
        lambda_start = self.lambda_
        for i, parameters in param_combos.items():
            self.running_grid_search = True
            try_clear_output()
            display_or_print(self.grid_search_results_df())
            self.show_param_combos_completed()
            print()
            if self.current_grid_search()['results'][i] is not None:
                print(MESSAGE_SKIPPING_PARAMS.format(i=i))
                self.running_grid_search = False
                continue
            L = parameters['L']
            lambda_ = parameters['lambda']
            print('L =', L, '; lambda =', lambda_)
            start_time = time.time()
            hyp = self.restore(input, L=L, lambda_=lambda_)
            evaluator = FeatureRestorationEvaluator(
                ref,
                hyp,
                capitalization=False,
                feature_chars=' ',
                get_wer_info_on_init=False
            )
            prf = evaluator.get_prfs()[' ']
            time_taken = time.time() - start_time
            self.current_grid_search()['results'][i] = {
                'i': i, 'L': L, 'lambda': lambda_,
                **prf, 'Time (s)': time_taken
            }
            self.restore_chunk.cache_clear()
            self.set_L(L_start)
            self.set_lambda(lambda_start)
            self.running_grid_search = False
            self.save()
        try_clear_output()
        display_or_print(self.grid_search_results_df())
        self.show_param_combos_completed()

    # ====================
    def grid_search_results_df(self) -> pd.DataFrame:
        """Get the results of the current grid search.

        Returns:
          pd.DataFrame
            A pandas dataframe containing the results for all the
            parameter combinations tested so far.
        """

        results = self.current_grid_search()['results'].copy()
        results = {k: v for k, v in results.items() if v is not None}
        if len(results) > 0:
            results_df = pd.DataFrame(results).transpose()
            results_df['i'] = results_df['i'].astype(int)
            results_df = results_df.set_index('i')
            results_df['L'] = results_df['L'].astype(int)
        else:
            results_df = pd.DataFrame()
        return results_df

    # ====================
    def param_combos_completed(self) -> Tuple[int, int]:
        """Get information about parameter combinations completed
        in the current grid search.

        Returns:
          Tuple[int, int]:
            The first element is the number of parameter combinations
            for which testing has been completed.
            The second element is the total number of parameter
            combinations
        """

        results = self.current_grid_search()['results'].copy()
        completed = len([r for r in results.values() if r is not None])
        total = len(results.keys())
        return completed, total

    # ====================
    def optimal_params(self,
                       metric_to_optimize: Optional[str] = None,
                       min_or_max: Optional[str] = None
                       ) -> Tuple[pd.DataFrame, Tuple[int, int]]:

        df = self.grid_search_results_df()
        self.set_metric_to_optimize(metric_to_optimize)
        self.set_min_or_max(min_or_max)
        metric_vals = df[self.metric_to_optimize].to_list()
        if self.min_or_max == 'max':
            optimal_val = max(metric_vals)
        elif self.min_or_max == 'min':
            optimal_val = min(metric_vals)
        optimal_rows = df[df[self.metric_to_optimize] == optimal_val]
        # There may be multiple optimal rows. Just return L and lambda
        # from the first optimal row.
        optimal_rows = optimal_rows.reset_index()
        L = int(optimal_rows.iloc[0]['L'])
        lambda_ = float(optimal_rows.iloc[0]['lambda'])
        return optimal_rows, (L, lambda_)

    # ====================
    def show_optimal_params(self,
                            metric_to_optimize: Optional[str] = None,
                            min_or_max: Optional[str] = None):
        """Display the rows from the grid search results table with the best
        results based on the values of the metric_to_optimize and min_or_max
        attributes of the class instance, and the values of the hyperparameters
        that produce those results.
        If there is more than one hyperparameter combination that produces the
        best result for metric_to_optimize, the one that was tested first will
        be selected.

        Args:
          metric_to_optimize (Optional[str], optional):
            If provided, the metric_to_optimize attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values.
            Defaults to None.
          min_or_max (Optional[str], optional):
            If provided, the min_or_max attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values. Defaults to None.
        """

        self.set_metric_to_optimize(metric_to_optimize)
        self.set_min_or_max(min_or_max)
        df, params = self.optimal_params()
        display_or_print(df)
        df = df.reset_index()
        L, lambda_ = params
        print(MESSAGE_OPTIMAL_PARAMS.format(
            L=L,
            lambda_=lambda_
        ))

    # ====================
    def set_optimal_params(self,
                           metric_to_optimize: Optional[str] = None,
                           min_or_max: Optional[str] = None):
        """Set the L and lambda_ attributes of the class instance to the
        optimal hyperparameters for the model based on the values of the
        metric_to_optimize and min_or_max attributes of the class instance.
        If there is more than one hyperparameter combination that produces the
        best result for metric_to_optimize, the one that was tested first will
        be selected.

        Args:
          metric_to_optimize (Optional[str], optional):
            If provided, the metric_to_optimize attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values.
            Defaults to None.
          min_or_max (Optional[str], optional):
            If provided, the min_or_max attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values. Defaults to None.
        """

        _, params = self.optimal_params(metric_to_optimize, min_or_max)
        L, lambda_ = params
        self.set_L(L)
        self.set_lambda(lambda_)
        self.save()

    # ====================
    def show_param_combos_completed(self):

        completed, total = self.param_combos_completed()
        print(MESSAGE_TESTED_SO_FAR.format(
            completed=completed,
            total=total
        ))

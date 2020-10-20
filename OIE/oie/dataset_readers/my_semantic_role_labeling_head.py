import logging
from typing import Dict, List, Iterable, DefaultDict, Optional, Iterator, Set, Tuple
from collections import defaultdict
import codecs
import os
from nltk import Tree
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("my-srl-head")
class MySrlReaderHead(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                head_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, head_label, tags)
            else:
                head_indicator = [1 if head == "H-V" else 2 if head == "H-A" else 0 for head in sentence.predicate_lemmas]
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    yield self.text_to_instance(tokens, verb_indicator, head_indicator, tags)

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         head_label: List[int] = None,
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        
        if head_label:
            fields['head_indicator'] = SequenceLabelField(head_label, text_field)
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)

        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens],
                                            "verb": verb})
        return Instance(fields)
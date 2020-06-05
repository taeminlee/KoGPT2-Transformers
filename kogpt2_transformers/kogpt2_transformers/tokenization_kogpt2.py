# python
import os
import json
# 3rd-party
from transformers.tokenization_utils import PreTrainedTokenizerFast
from transformers.file_utils import hf_bucket_url, cached_path
from tokenizers import SentencePieceBPETokenizer

class KoGPT2TokenizerFast(PreTrainedTokenizerFast):
    """
    Constructs a "Fast" GPT-2 BPE tokenizer (backed by HuggingFace's `tokenizers` library).
    Peculiarities:
    - Byte-level Byte-Pair-Encoding
    - Requires a space to start the input string => the encoding methods should be called with the
      ``add_prefix_space`` flag set to ``True``.
      Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
      the absence of a space at the beginning of a string:
    ::
        tokenizer.decode(tokenizer.encode("Hello")) = " Hello"
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        errors (:obj:`str`, `optional`, defaults to "replace"):
            Paradigm to follow when decoding bytes to UTF-8. See `bytes.decode
            <https://docs.python.org/3/library/stdtypes.html#bytes.decode>`__ for more information.
        unk_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (:obj:`bool`, `optional`, defaults to `False`):
            Whether to add a leading space to the first word.
            This allows to treat the leading word just as any other word.
            (GPT2 tokenizer detect beginning of words by the preceeding space)
        trim_offsets (:obj:`bool`, `optional`, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "special_tokens_map_file": "special_tokens_map.json",
        "added_tokens_file": "added_tokens.json"
    }

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_prefix_space=False,
        **kwargs
    ):
        super().__init__(
            SentencePieceBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                add_prefix_space=add_prefix_space,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )
    
    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        tok = cls._from_pretrained(*inputs, **kwargs)
        tok.add_tokens(list(tok.unique_added_tokens_encoder))
        pretrained_model_name_or_path = inputs[0]
        if os.path.isdir(pretrained_model_name_or_path):
            full_file_name = os.path.join(pretrained_model_name_or_path, cls.vocab_files_names['special_tokens_map_file'])
            if not os.path.exists(full_file_name):
                full_file_name = None
        else:
            full_file_name = hf_bucket_url(
                pretrained_model_name_or_path, filename=cls.vocab_files_names['special_tokens_map_file'], use_cdn=False
            )
        
        if(full_file_name is not None):
            with open(cached_path(full_file_name)) as fp:
                special_tokens = json.load(fp)
            tok.add_special_tokens(special_tokens)
        return tok
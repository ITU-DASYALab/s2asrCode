
To use the language model you have to either make it yourself or download one, such as the one on Mozilla's DeepSpeech repository.

Credit Mozilla DeepSpeech. https://github.com/mozilla/DeepSpeech

The tests we have conducted seems to indicate that the word based beam search gives better WERs for our purposes.

Or make your own:

´´´bash

   import gzip
   import io
   import os

   from urllib import request

   # Grab corpus.
   url = 'http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz'
   data_upper = '/tmp/upper.txt.gz'
   request.urlretrieve(url, data_upper)

   # Convert to lowercase and cleanup.
   data_lower = '/tmp/lower.txt'
   with open(data_lower, 'w', encoding='utf-8') as lower:
       with io.TextIOWrapper(io.BufferedReader(gzip.open(data_upper)), encoding='utf8') as upper:
           for line in upper:
               lower.write(line.lower())

   # Build pruned LM.
   lm_path = '/tmp/lm.arpa'
   !lmplz --order 5 \
          --temp_prefix /tmp/ \
          --memory 50% \
          --text {data_lower} \
          --arpa {lm_path} \
          --prune 0 0 0 1

   # Quantize and produce trie binary.
   binary_path = '/tmp/lm.binary'
   !build_binary -a 255 \
                 -q 8 \
                 trie \
                 {lm_path} \
                 {binary_path} 
   os.remove(lm_path)

The trie was then generated from the vocabulary of the language model:
´´´
### neural-wsd related:
    - osman:
        - [big] predict all the sentence by using *token embeddings* with RNN.
        - logger doesn't fucking work.
        - [big] create a very small roberta to do basic functionality tets.
        - Documentation for critical methods.
        - how to save network partially - only the trained weights?
        - loss calc looks wrong.
        - [big] add some sanity checks that everything works correctly.
        - [big] a general tensorboard integration.
        - splitting data
            - perhaps instead of having one global loss, let's have array of losses. 
### offline-wiki related:
    - kerem
        - data analysis - how long are the sentences? Some plots would be good. Perhaps a barplot (x axis target words, y axis avg. # of words for a target word)
        - Linkpages
        - Word distribution of Wiki - create a dictionary of word and its occurrence. Use whole wikipedia and print a file in a format (word<tab>occurence).

### done
- [big] predict only the sense of a target word by using *token* embedding.
- restructuring the project.
- [big] divide the model - pretrained and basic nn - ``RobertaTokenModel`` added in ``model/transformer_based_models.py``
- test WordPieceToTokenList Transformer
- Runner options increased. 
- Implement all the functionality to use one token (not [CLS]) to do classification.
- Refactoring for model -> base.py
- Run the model with the first token - (not the [CLS] token)
- adding wordpieceToToken transformer
- improved setup process & pre-commit stuff.
- add asciification to transformation pipeline.
- refactoring training
- change batch size for evaluation.
- adding validation part.
- improved run script.

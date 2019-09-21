### neural-wsd related:
    - osman:
        - [big] divide the model - pretrained and basic nn
        - restructuring the project.
        - how to save network partially - only the trained weights?
        - loss calc looks wrong.
        - test WordPieceToTokenList Transformer
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
- improved setup process & pre-commit stuff.
- add asciification to transformation pipeline.
- refactoring training
- change batch size for evaluation.
- adding validation part.
- improved run script.
- adding wordpieceToToken transformer

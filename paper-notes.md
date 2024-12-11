# KBQA

## SP-based Methods in Paper KQA Pro

Semantic Parsing Based Methods: parses a question to a symbolic logic form, and then executes it against the KB and obtains the final answers.

Paper: [KQA Pro: A Dataset with Explicit Compositional Programs for Complex Question Answering over Knowledge Base](https://arxiv.org/abs/2007.03875)

GitHub Repository: [https://github.com/shijx12/KQAPro_Baselines](https://github.com/shijx12/KQAPro_Baselines)

### Concepts

- **KoPL**: Knowledge-oriented Programming Language proposed by the paper, to describe the reasoning process for solving complex questions (see [Fig. 1](https://ar5iv.org/html/2007.03875#S1.F1))

- **Triples**: (canonical question, KoPL, SPARQL)

- **KQA Pro**: 

  |                 | multiple kinds of knowledge | number of questions | natural language | query graphs | multi-step programs |
  | :-------------: | :-------------------------: | :-----------------: | :--------------: | :----------: | :-----------------: |
  | KQA Pro Dataset |              ✔              |       117,970       |        ✔         |      ✔       |          ✔          |

- **Weak Supervision Signals**: only question-answer pairs

- **NLQs**: natural language questions

- **KB Structure**: see [section 3.1](https://ar5iv.org/html/2007.03875#S3.SS1)

- **Multi-hop**: refers to multi-hop questions (check KoPL output for visualization)

### Dataset Formats

- train/valid/test sizes: 94,376/11,797/11,797

- `train.json` and `val.json` format:

  - **5 elements**: question, SPARQL query, KoPL program, 10 answer choices, and a golden answer

  - Choices are selected by executing an abridged SPARQL, which randomly drops one clause from the complete SPARQL
  - Supports both multiple-choice setting and open-ended setting

  ```json
  {
    "question": "Who was the prize winner when Mrs. Miniver got the Academy Award for Best Writing, Adapted Screenplay?",
    "choices": [
      "Canada–Hong Kong relations",
      "Bolivia–Brazil border",
      "15th Academy Awards",
      "South Sudan–United States relations",
      "Australia–Netherlands relations",
      "Ghana–Russia relations",
      "Azerbaijan–China relations",
      "25th Tony Awards",
      "Chad–Sudan border",
      "Egypt–Indonesia relations"
    ],
    "program": [],  // KoPL program, omitted
    "sparql": "SELECT DISTINCT ?qpv WHERE { ?e_1 <pred:name> \"Mrs. Miniver\" . ?e_2 <pred:name> \"Academy Award for Best Writing, Adapted Screenplay\" . ?e_1 <award_received> ?e_2 . [ <pred:fact_h> ?e_1 ; <pred:fact_r> <award_received> ; <pred:fact_t> ?e_2 ] <statement_is_subject_of> ?qpv .  }",
    "answer": "15th Academy Awards"
  }
  ```

- `test.json` format:

  - **2 elements**: question, 10 answer choices

  ```json
  {
    "question": "Which movie is shorter, Tron: Legacy or Pirates of the Caribbean: The Curse of the Black Pearl?",
    "choices": [
      "Pirates of the Caribbean: The Curse of the Black Pearl",
      "The Flintstones in Viva Rock Vegas",
      "Holes",
      "National Treasure: Book of Secrets",
      "Tron: Legacy",
      "Old Dogs",
      "G-Force",
      "The Muppets",
      "Alvin and the Chipmunks: Chipwrecked",
      "John Carter"
    ]
  }
  ```


### Models used in Paper

Illustrations for the models in the code repository: 

```bash
.
├── Bart_Program  # SP-based model, generates KoPL
├── Bart_SPARQL  # SP-based model, generates SPARQL  !!!!!!
├── Program  # SP-based model, generates KoPL
├── SPARQL  # SP-based model, generates SPARQL  !!!!!!
├── RGCN  # Other baseline model
├── SRN  # Other baseline model
├── KVMemNN  # Other baseline model
├── BlindGRU  # predicts answer with only the input question, ignoring the knowledge base
```

The models(directories) we care for the NLP project: **Bart_SPARQL** and **SPARQL** directories

SP-based Models: 

- sequence-to-sequence model—**RNN** with attention mechanism: generate **SPARQL query** and **KoPL program** at the same time
- pretrained generative language model—**BART** (bart-base from HuggingFace ):  same as above

Other models (unrelated to SP-based methods): KVMemNet, EmbedKGQA , SRN, RGCN




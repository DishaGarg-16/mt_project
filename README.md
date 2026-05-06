# English to Hindi Machine Translation: Does More (Synthetic) Data Help?

## Table of Contents

- [Idea and Motivation](#idea-and-motivation)
- [Project Architecture](#project-architecture)
  - [Phase 1 -- Data Acquisition and Cleaning](#phase-1----data-acquisition-and-cleaning)
  - [Phase 2 -- Baseline Training](#phase-2----baseline-training)
  - [Phase 3 -- Synthetic Data Augmentation via LLM Paraphrasing](#phase-3----synthetic-data-augmentation-via-llm-paraphrasing)
  - [Phase 4 -- Augmented Training and Comparison](#phase-4----augmented-training-and-comparison)
- [The Transformer Architecture and How It Is Used](#the-transformer-architecture-and-how-it-is-used)
  - [Encoder-Decoder Transformers for Machine Translation](#encoder-decoder-transformers-for-machine-translation)
  - [Helsinki-NLP/opus-mt-en-hi -- The Pre-trained Model](#helsinki-nlpopus-mt-en-hi----the-pre-trained-model)
  - [How the Transformer Is Applied in This Project](#how-the-transformer-is-applied-in-this-project)
- [Iterative Development Process](#iterative-development-process)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Improvement -- Cosine Similarity of Embeddings](#future-improvement----cosine-similarity-of-embeddings)
  - [The Problem It Would Solve](#the-problem-it-would-solve)
  - [How It Would Work](#how-it-would-work)
  - [Expected Impact](#expected-impact)

---

## Idea and Motivation

Machine translation systems require large amounts of parallel data (paired sentences in both source and target languages) to perform well. In low-resource settings, where only a small number of parallel pairs are available, translation quality suffers significantly. This project investigates a practical question: **can synthetic data augmentation through LLM-based paraphrasing improve the performance of a neural machine translation model when the original dataset is small?**

The core idea is straightforward. Take a small, cleaned subset of English-Hindi parallel data, train a baseline translation model on it, then generate additional training pairs by paraphrasing the English side of existing pairs using a large language model (while keeping the Hindi targets unchanged), and finally retrain the model on the combined dataset. If the augmented model outperforms the baseline on a held-out test set, it validates the hypothesis that LLM-generated paraphrases can serve as a cost-effective augmentation strategy.

The experiment is designed as a controlled A/B comparison. Both the baseline and augmented models are trained with the same total number of training pairs (3000), the same hyperparameters, and evaluated on the same held-out test set. The only difference is the composition of the training data: the baseline duplicates 1000 samples from its own training set, while the augmented model replaces those duplicates with 1000 LLM-paraphrased English sentences paired with their original Hindi translations.

---

## Project Architecture

The project follows a four-phase pipeline, developed iteratively over multiple rounds of experimentation.

### Phase 1 -- Data Acquisition and Cleaning

Data is sourced from the **IIT Bombay English-Hindi Parallel Corpus** (`cfilt/iitb-english-hindi`), a large-scale publicly available dataset containing over 1.6 million sentence pairs. From this corpus, 20,000 pairs are randomly sampled (seed=42), then subjected to a cleaning pipeline:

1. **Null removal** -- Drop any pair where either the English or Hindi side is empty.
2. **Whitespace normalization** -- Strip leading and trailing whitespace from both sides.
3. **Minimum length filter** -- Remove pairs where either side contains fewer than 4 words (to exclude fragments, titles, and noisy short strings).
4. **Pattern-based noise filter** -- Exclude pairs containing digits, curly braces, dollar signs, or percent signs (to remove formulaic, code-like, or tabular content).

After cleaning, 2500 pairs are selected from the filtered pool and split deterministically:

- **Train**: 2000 pairs
- **Validation**: 250 pairs (used for early stopping during training)
- **Test**: 250 pairs (held out for final evaluation only)

This data-cleaning step was a critical lesson learned during the iterative development. Early iterations skipped the cleaning phase and produced poor results because the dataset contained noisy, short, or numerically-heavy sentences that degraded both training and evaluation quality.

### Phase 2 -- Baseline Training

The baseline model is built by fine-tuning the pre-trained `Helsinki-NLP/opus-mt-en-hi` transformer model. To create a fair comparison (same training set size as the augmented model), 1000 sentences are randomly sampled from the 2000 training pairs and duplicated, bringing the baseline training set to 3000 pairs.

Training uses the Hugging Face `Seq2SeqTrainer` with the following configuration:

- **Epochs**: Up to 10 (with early stopping, patience=2)
- **Batch size**: 16
- **Learning rate**: 3e-5
- **Weight decay**: 0.01
- **Evaluation strategy**: Per-epoch on the validation set
- **Best model selection**: Based on BLEU score
- **Mixed precision (FP16)**: Enabled when GPU is available

Early stopping monitors BLEU on the validation set and halts training if no improvement occurs for 2 consecutive epochs. This prevents overfitting, which was a significant problem observed in early iterations when all 10 epochs were run unconditionally.

### Phase 3 -- Synthetic Data Augmentation via LLM Paraphrasing

From the 2000 training pairs, 1000 are randomly selected (fixed seed for reproducibility) and exported to a CSV file. An external script (`generate_paraphrases.py`) then processes these sentences through the **Gemini API** to produce paraphrased versions of the English sentences. The Hindi translations remain unchanged.

The paraphrasing approach preserves semantic meaning while introducing lexical and syntactic variation. For example, "I am going to school" might become "I am heading to school" or "I will be attending school." The model then encounters multiple English surface forms that all map to the same Hindi translation, which in theory should broaden its understanding of English input diversity.

Key implementation details of the paraphrasing pipeline:

- **Checkpointing**: The script saves progress periodically, allowing it to resume after API rate limits or failures.
- **Batch processing**: Sentences are processed in batches to manage API quotas efficiently.
- **Quality control**: The resulting CSV is loaded back into the notebook, and samples are manually inspected before use.

### Phase 4 -- Augmented Training and Comparison

The augmented training set is constructed by combining the original 2000 training pairs with the 1000 paraphrased pairs (paraphrased English + original Hindi), yielding 3000 total training pairs. The augmented model is then trained from scratch using the exact same hyperparameters as the baseline.

Both models are evaluated on the same held-out test set using BLEU, ROUGE-1, ROUGE-L, and chrF metrics. Additionally, qualitative comparisons are performed by translating 15 sample test sentences through both models and comparing outputs side-by-side.

---

## The Transformer Architecture and How It Is Used

### Encoder-Decoder Transformers for Machine Translation

The transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), replaced recurrent neural networks as the dominant approach for sequence-to-sequence tasks like machine translation. A transformer consists of two main components:

**The Encoder** takes the source sentence (English) and processes it through multiple layers of self-attention and feed-forward networks. Each self-attention layer allows every token in the input to attend to every other token, capturing long-range dependencies and contextual relationships. The output is a sequence of context-rich hidden representations, one per input token. These representations encode not just the meaning of individual words but also their roles and relationships within the sentence.

**The Decoder** generates the target sentence (Hindi) one token at a time. It uses two types of attention: masked self-attention (which prevents the decoder from looking ahead at future tokens it has not yet generated) and cross-attention (which allows each decoder position to attend to the full encoder output). This cross-attention mechanism is the bridge between the two languages -- it lets the decoder "look at" the entire source sentence when deciding what to generate next.

Both encoder and decoder use **positional encodings** to inject information about token order, since attention mechanisms are inherently order-agnostic. Multi-head attention allows the model to simultaneously attend to information from different representation subspaces, capturing different types of linguistic relationships.

### Helsinki-NLP/opus-mt-en-hi -- The Pre-trained Model

This project uses `Helsinki-NLP/opus-mt-en-hi`, a Marian-based encoder-decoder transformer specifically pre-trained for English-to-Hindi translation. Marian NMT is a C++-based framework optimized for fast training and translation, and the Hugging Face implementation wraps these models in the standard `transformers` API.

Key characteristics of this model:

- It is a **6-layer encoder, 6-layer decoder** transformer based on the MarianMT architecture.
- It was pre-trained on the OPUS parallel corpus collection, which aggregates multiple English-Hindi parallel datasets.
- The tokenizer uses **SentencePiece**, a subword tokenization algorithm that handles morphologically rich languages (like Hindi) effectively by breaking words into smaller, meaningful subword units.
- The model uses **shared embeddings** between encoder and decoder, meaning both sides share the same vocabulary and embedding space.

### How the Transformer Is Applied in This Project

Rather than training a transformer from scratch (which would require millions of sentence pairs and significant compute), this project **fine-tunes** the pre-trained model on a small domain-specific dataset. Fine-tuning adjusts the model's existing weights to better handle the particular distribution of sentences in the IIT Bombay corpus subset.

During fine-tuning:

1. **Tokenization**: Both English and Hindi sentences are tokenized into subword sequences using the pre-trained SentencePiece tokenizer, with a maximum sequence length of 128 tokens. Padding tokens are masked with -100 so the loss function ignores them.

2. **Forward pass**: The English tokens are fed through the encoder, producing contextual representations. The decoder then generates Hindi tokens autoregressively, using teacher forcing (the ground-truth Hindi tokens are provided as decoder input during training).

3. **Loss computation**: Cross-entropy loss is computed between the predicted token probabilities and the ground-truth Hindi tokens. Padding positions are excluded.

4. **Optimization**: The AdamW optimizer updates all model parameters with a learning rate of 3e-5 and weight decay of 0.01. FP16 mixed precision is used for faster training on GPU.

5. **Inference**: During evaluation and translation, the model uses **beam search** (via `model.generate()`) to produce translations, generating tokens one at a time until an end-of-sequence token is produced.

The core hypothesis of this project is that by exposing the encoder to more diverse English phrasings (via paraphrasing) that map to the same Hindi translations, the encoder's representations become more robust and generalize better to unseen test inputs.

---

## Iterative Development Process

This project was not built in a single pass. It evolved through multiple iterations, each addressing problems discovered in the previous round. The full development arc, as documented in the project's design diagram, is summarized below:

**Iteration 1 -- Initial Experimentation**: The first attempt trained the model for a fixed 10 epochs on a small sample, which resulted in severe overfitting. The training loss dropped steadily while the validation loss plateaued or increased after just a few epochs. This led to the implementation of early stopping with a patience of 2 epochs.

**Iteration 2 -- Baseline Training and API Limitations**: A baseline model was trained on 2000 pairs. An attempt was made to generate 1000 paraphrases using the Gemini API's free tier, but rate limits prevented completion. This prompted the shift to an offline, checkpoint-based paraphrasing script.

**Iteration 3 -- Dataset Refinement**: Earlier iterations used random subsets without a fixed seed, making results non-reproducible. This was corrected by fixing the random seed at 42 and establishing a formal train/validation/test split (2000/250/250).

**Iteration 4 -- Data Cleaning**: The most impactful change came when the data cleaning pipeline was introduced. Previous iterations included noisy sentences with numbers, special characters, and very short fragments. Filtering these out significantly improved both baseline and augmented model performance -- BLEU scores jumped from around 10 to above 30.

**Final Iteration -- Controlled Comparison**: The final experiment used cleaned data, proper splits, early stopping, and a carefully constructed baseline (2000 originals + 1000 duplicates) versus augmented (2000 originals + 1000 paraphrased) training set, both totaling 3000 pairs.
![Description](iterations.png)
---

## Results

The final test set comparison between the two models:

| Metric  | Baseline (3000) | Augmented (3000) | Delta |
|---------|-----------------|------------------|-------|
| BLEU    | 30.72           | 30.77            | +0.05 |
| ROUGE-1 | 0.0232          | 0.0232           |  0.00 |
| ROUGE-L | 0.0232          | 0.0232           |  0.00 |
| chrF    | 45.37           | 45.36            | -0.01 |

Both models stopped training at epoch 5 (out of 10) due to early stopping, confirming that additional epochs would have led to overfitting.

The augmented model showed a marginal improvement in BLEU (+0.05) and essentially identical performance on other metrics. Qualitative analysis of 15 test translations revealed that both models produced identical outputs for all samples, suggesting that the paraphrasing-based augmentation had minimal impact at this scale.

This near-zero improvement likely stems from several factors:
- The paraphrased English sentences may not have been sufficiently different from the originals.
- With only 1000 paraphrased samples (out of 3000 total), the signal from augmentation was diluted.
- The pre-trained model already has strong English-language understanding from its original OPUS training data, so additional English diversity provides diminishing returns.

---

## Evaluation Metrics

- **BLEU (Bilingual Evaluation Understudy)**: Measures n-gram overlap between the generated translation and reference translation. Scores range from 0 to 100, with higher scores indicating closer matches. It is the most widely used metric for machine translation.

- **ROUGE-1**: Measures unigram (single-word) overlap between the generated and reference texts. Originally designed for summarization evaluation.

- **ROUGE-L**: Measures the longest common subsequence between generated and reference texts, capturing sentence-level structure similarity.

- **chrF (Character F-score)**: Computes character-level n-gram F-score between generated and reference translations. Particularly useful for morphologically rich languages like Hindi, where word-level metrics may miss partial matches at the character level.

---

## Future Improvement -- Cosine Similarity of Embeddings

One significant limitation of the current approach is the absence of any quality control mechanism for the paraphrased sentences. The LLM-generated paraphrases are used as-is, without verifying whether they genuinely preserve the semantic meaning of the original English sentence. This is a critical gap -- a paraphrase that drifts too far from the original meaning will create a misaligned training pair (where the English sentence no longer accurately corresponds to the Hindi translation), injecting noise into the training data. Conversely, a paraphrase that is too similar to the original (differing by only a word or two) provides minimal new information and is effectively a near-duplicate.

### The Problem It Would Solve

Cosine similarity of sentence embeddings would provide a quantitative measure of semantic similarity between the original English sentence and its paraphrased version. This enables two types of filtering:

1. **Filtering out bad paraphrases**: If the cosine similarity between the original and paraphrased embeddings is below a certain threshold (for example, 0.7), the paraphrase has likely drifted too far from the original meaning and should be discarded. Using such a pair for training would teach the model incorrect mappings.

2. **Filtering out trivial paraphrases**: If the cosine similarity is above a very high threshold (for example, 0.98), the paraphrase is essentially identical to the original and adds no meaningful diversity to the training data. These near-duplicates waste training capacity without providing new signal.

### How It Would Work

The implementation would involve:

1. **Embedding generation**: Use a pre-trained sentence embedding model (such as `sentence-transformers/all-MiniLM-L6-v2` or a multilingual variant) to encode both the original English sentence and its paraphrased version into fixed-dimensional vector representations.

2. **Cosine similarity computation**: For each original-paraphrase pair, compute the cosine similarity between their embedding vectors. Cosine similarity ranges from -1 (opposite meaning) to 1 (identical meaning), with values near 1 indicating high semantic similarity.

3. **Band-pass filtering**: Retain only paraphrases whose cosine similarity falls within a target range (for example, 0.75 to 0.95). This ensures the paraphrase is semantically faithful enough to preserve the alignment with the Hindi translation, yet lexically and syntactically different enough to provide genuine augmentation value.

4. **Iterative regeneration**: For paraphrases that fall outside the acceptable range, re-prompt the LLM with adjusted instructions (for example, requesting a more conservative or more creative paraphrase) and re-evaluate.

### Expected Impact

Introducing cosine similarity filtering would address one of the key weaknesses in the current pipeline: the lack of quality assurance for synthetic training data. By ensuring that every paraphrased pair is both semantically accurate and meaningfully different, the augmented dataset would be of higher quality. This should lead to a more noticeable improvement in translation metrics, as the model would receive genuinely diverse yet correctly-aligned training signal rather than a mix of near-duplicates and potentially misaligned pairs.

This improvement would also make the augmentation pipeline more scalable. Rather than manually inspecting samples, the cosine similarity scores provide an automated, reproducible quality gate that can be applied to thousands or tens of thousands of paraphrased sentences.

# Awesome-Pathology-Agents

Awesome papers and datasets specifically focused on pathology.

- [Awesome-Pathology-Agents](#awesome-pathology-agents)
  - [Papers](#papers)
    - [Perception](#perception)
    - [Reasoning](#reasoning)
    - [Planning](#planning)
    - [Tool Use](#tool-use)
    - [Memory](#memory)
    - [Self-improvement](#self-improvement)
  - [Datasets](#datasets)
    - [Patch Level Visual Question Answering Datasets](#patch-level-visual-question-answering-datasets)
    - [Slide Level Visual Question Answering Datasets](#slide-level-visual-question-answering-datasets)
  - [Star History](#star-history)

## Papers

### Perception

* Quilt-1M: One Million Image-Text Pairs for Histopathology, NIPS 2023.
* PathAlign: A vision–language model for whole slide images in histopathology, MICCAI COMPAYL Workshop 2024.
* WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole-Slide Images, MICCAI 2024 oral.
* HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-Modal Context Interaction, MICCAI 2024.
* A multimodal generative AI copilot for human pathology, Nature 2024.
* Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos, CVPR 2024.
* A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis, arXiv 2024.
* PRISM2: Unlocking Multi-Modal General Pathology AI with Clinical Dialogue, arXiv 2025.
* Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology, arXiv 2025.
* ALPaCA: Adapting Llama for Pathology Context Analysis to enable slide-level question answering, medrxiv 2025.
* PathVG: A New Benchmark and Dataset for Pathology Visual Grounding, MICCAI 2025.
* A vision–language foundation model for precision oncology, Nature 2025.
* A multimodal whole-slide foundation model for pathology, Nature Medicine 2025.
* Generating dermatopathology reports from gigapixel whole slide images with HistoGPT, Narture Communications 2025.
* WSI-LLaVA: A Multimodal Large Language Model for Whole Slide Image, ICCV 2025.

### Reasoning

* Evidence-based diagnostic reasoning with multi-agent copilot for human pathology, arXiv 2025.
* A Versatile Pathology Co-pilot via Reasoning Enhanced Multimodal Large Language Model, arXiv 2025.
* Pathology-CoT: Learning Visual Chain-of-Thought Agent from Expert Whole Slide Image Diagnosis Behavior, arXiv 2025.
* TeamPath: Building MultiModal Pathology Experts with Reasoning AI Copilots, arXiv 2025.
* Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner, AAAI 2026.
* CPath-Omni: A Unified Multimodal Foundation Model for Patch and Whole Slide Image Analysis in Computational Pathology, CVPR 2025.
* SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding, CVPR 2025.

### Planning

* PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology, ICCV 2025.
* PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning, arXiv 2025.

### Tool Use

* PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology, AAAI 2024.
* WSI-Agents: A Collaborative Multi-agent System for Multi-modal Whole Slide Image Analysis, MICCAI 2025.
* PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration, ICLR 2025 oral.
* UnPuzzle: A Unified Framework for Pathology Image Analysis, arXiv 2025.

### Memory

* PolyPath: Adapting a Large Multimodal Model for Multislide Pathology Report Generation, Modern Pathology 2025.
* SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction, arXiv 2025.
* Patho-AgenticRAG: Towards Multimodal Agentic Retrieval-Augmented Generation for Pathology VLMs via Reinforcement Learning, AAAI 2026.

### Self-improvement

* A co-evolving agentic AI system for medical imaging analysis, arXiv 2025.

### Other

* PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology, ECCV 2024, Benchmark.
* PathBench: Advancing the Benchmark of Large Multimodal Models for Pathology Image Understanding at Patch and Whole Slide Level, TMI 2025, Benchmark.
* Eye-Tracking, Mouse Tracking, Stimulus Tracking, and Decision-Making Datasets in Digital Pathology, arXiv 2025, Clinical hardware data collection.


## Datasets

***Latest Papers (after 2023)***
[ Dataset ] Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers, CVPR 2024.
[ Dataset ] MovieLLM: Enhancing Long Video Understanding with AI-Generated Movies, 2024.
  A novel framework designed to create synthetic, high-quality data for long videos. This framework leverages the power of GPT-4 and text-to-image models to generate detailed scripts and corresponding visuals.

### Patch Level Visual Question Answering Datasets

| Dataset              | Annotation                                      | Source               | Number             | Duration | Tasks                                   | link                                                                                                                                                | Date Released |
| -------------------- | ----------------------------------------------- | -------------------- | ------------------ | -------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| EgoLife | timestamps + captions | Daily Life | 6 | 44.3h | Understanding & Reasoning | [Data](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342), [Proj Page](https://egolife-ai.github.io/), [Paper](https://arxiv.org/abs/2503.03803) | 2025
| ActivityNet 1.3      | timestamps + action                             | Youtube              | 20k                | -        | Action Localization                     |                                                                                                                                                     |               |
| ActivityNet Captions | timestamps + captions                           | Youtube              | 20k                | -        | Dense captioning, video grounding       |                                                                                                                                                     |               |
| THUMOS               | timestamps + action                             | -                    | -                  | -        | Action Localization                     |                                                                                                                                                     |               |
| YouCook2             | timestamps + captions                           | Cooking Videos       | -                  | -        | Dense captioning                        |                                                                                                                                                     |               |
| MovieNet             | timestamps + captions + place/action/style tags | Movies               | 1.1k               | >2h      | movie understanding                     | [MovieNet](https://movienet.site/)                                                                                                                     | 2020          |

### Slide Level Visual Question Answering Datasets

| Dataset              | Annotation                                      | Source               | Number             | Duration | Tasks                                   | link                                                                                                                                                | Date Released |
| -------------------- | ----------------------------------------------- | -------------------- | ------------------ | -------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| EgoLife | timestamps + captions | Daily Life | 6 | 44.3h | Understanding & Reasoning | [Data](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342), [Proj Page](https://egolife-ai.github.io/), [Paper](https://arxiv.org/abs/2503.03803) | 2025
| ActivityNet 1.3      | timestamps + action                             | Youtube              | 20k                | -        | Action Localization                     |                                                                                                                                                     |               |
| ActivityNet Captions | timestamps + captions                           | Youtube              | 20k                | -        | Dense captioning, video grounding       |                                                                                                                                                     |               |
| THUMOS               | timestamps + action                             | -                    | -                  | -        | Action Localization                     |                                                                                                                                                     |               |
| YouCook2             | timestamps + captions                           | Cooking Videos       | -                  | -        | Dense captioning                        |                                                                                                                                                     |               |
| MovieNet             | timestamps + captions + place/action/style tags | Movies               | 1.1k               | >2h      | movie understanding                     | [MovieNet](https://movienet.site/)                                                                                                                     | 2020          |

## Benchmarks

***Latest Papers (after 2023)***
* [**TemporalBench**: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models],arXiv[![arXiv](https://img.shields.io/badge/arXiv-2410.10818-b31b1b.svg?style=plastic)](https://www.arxiv.org/abs/2410.10818)
* [**VideoHallucer**: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models](https://arxiv.org/abs/2406.16338), arXiv[![arXiv](https://img.shields.io/badge/arXiv-2406.16338-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2406.16338)
* Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis,2024
* CinePile: A Long Video Question Answering Dataset and Benchmark,2024
* TempCompass: Do Video LLMs Really Understand Videos? 2024.
* MVBench: A Comprehensive Multi-modal Video Understanding Benchmark, CVPR 2024.
* How Good is my Video LMM? Complex Video Reasoning and Robustness Evaluation Suite for Video-LMMs, 2024
* MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding [Arxiv](https://arxiv.org/abs/2406.04264)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=G14nTDo4/PathAgent&type=date&legend=top-left)](https://www.star-history.com/#G14nTDo4/PathAgent&type=date&legend=top-left)

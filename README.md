# MultihopQA-Reasoning-Retrieve-Survey
# Explaining Multi-hop Question Answering via Retrieval–Reasoning Processes

A paper list curated for our survey on **retrieval–reasoning procedures** in multi-hop QA / RAG.  
This README is organized **by the same “cases” used in the LaTeX paper** (i.e., the bold subcategories under each design axis).

## Contents
- [T4: Retrieval–Reasoning Coupling (Design Axes)](#t4-retrievalreasoning-coupling-design-axes)
  - [A. Overall Execute Plan](#a-overall-execute-plan)
  - [B. Index Structure](#b-index-structure)
  - [C. Next-Step Execute Plan](#c-next-step-execute-plan)
  - [D. Stop / Continue Criteria](#d-stop--continue-criteria)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Surveys & Meta](#surveys--meta)
- [Contributing](#contributing)

---

## T4: Retrieval–Reasoning Coupling (Design Axes)

### A. Overall Execute Plan

#### Retrieve–then–Read
- (arXiv 2017) **Reading wikipedia to answer open-domain questions** [[Paper]](https://arxiv.org/abs/1704.00051)
- (EMNLP 2020) **Dense Passage Retrieval for Open-Domain Question Answering**
- (EMNLP 2021) **Leveraging passage retrieval with generative models for open domain question answering**
- (EMNLP 2020) **Retrieval-augmented generation for knowledge-intensive nlp tasks**
- (EMNLP-IJCNLP 2019) **Answering complex open-domain questions through iterative query generation**

#### Interleaved Retrieval and Reasoning
- (Findings 2023) **Measuring and Narrowing the Compositionality Gap in Language Models**
- (ICLR 2022) **React: Synergizing reasoning and acting in language models**
- (ACL 2023) **Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions**
- (arXiv 2023) **Active retrieval augmented generation**
- (ACL 2024) **DRAGIN: Dynamic retrieval augmented generation based on the real-time information needs of large language models** [[Paper]](https://aclanthology.org/2024.acl-long.743/)

#### Plan–then–Execute
- (ICLR 2022) **Decomposed prompting: A modular approach for solving complex tasks**
- (EMNLP 2023) **Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models**
- (NeurIPS 2022) **Least-to-most prompting enables complex reasoning in large language models**
- (EMNLP 2020) **Break it down: A question decomposition dataset for complex question answering**
- (arXiv 2025) **Credible plan-driven RAG method for multi-hop question answering (PAR RAG)**
- (ACL 2024) **End-to-End Beam Retrieval for Multi-Hop Question Answering** [[Paper]](https://aclanthology.org/2024.acl-long.315/)

#### Test-Time Search Scaling
- (NeurIPS 2023) **Tree of thoughts: Deliberate problem solving with large language models**
- (ICLR 2024) **Graph of thoughts: Solving elaborate problems with large language models**
- (arXiv 2024) **MindStar: Enhancing reasoning with large language models using search**
- (arXiv 2024) **Monte Carlo tree search for reasoning with language models**
- (SIGIR 2025) **Rearter: Retrieval-augmented reasoning with trustworthy process rewarding**
- (EMNLP 2023) **Reasoning with language model is planning with world model**
- (arXiv 2024) **Retrieval-augmented generation with graphs (graphrag)** [[Paper]](https://arxiv.org/abs/2501.00309)

---

### B. Index Structure

#### Flat / Candidate-List Indices
- (arXiv 2017) **Reading wikipedia to answer open-domain questions** [[Paper]](https://arxiv.org/abs/1704.00051)
- (EMNLP 2020) **Dense Passage Retrieval for Open-Domain Question Answering**
- (EMNLP 2021) **Leveraging passage retrieval with generative models for open domain question answering**
- (EMNLP 2020) **Answering complex open-domain questions through iterative query generation**
- (AAAI 2021) **HopRetriever: Retrieve hops over wikipedia to answer complex questions**
- (ACL 2023) **Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions**
- (TACL 2022) **MuSiQue: Multi-hop questions via single-hop question composition**
- (NAACL 2025) **Resource-Friendly Dynamic Enhancement Chain for Multi-hop QA (DEC)** [[Paper]](https://aclanthology.org/2025.naacl-long.650/)
- (ACL 2024) **End-to-End Beam Retrieval for Multi-Hop Question Answering** [[Paper]](https://aclanthology.org/2024.acl-long.315/)
- (arXiv 2025) **Credible plan-driven RAG method for multi-hop question answering (PAR RAG)**
- (ICLR 2024) **Raptor: Recursive abstractive processing for tree-organized retrieval**
- (arXiv 2024) **LongRAG: Leveraging Long Context to Enhance Retrieval-Augmented Generation**
- (arXiv 2024) **LongRAG: A Dual-Stage Retrieval Framework for Long Context**
- (NAACL 2025) **SIM-RAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains** [[Paper]](https://aclanthology.org/2025.naacl-long.575/)
- (arXiv 2025) **Stop-RAG: Value-Based Retrieval Control for Iterative RAG** [[Paper]](https://arxiv.org/abs/2510.14337)

#### Hierarchical / Summary-Tree Indices
- (ICLR 2024) **Raptor: Recursive abstractive processing for tree-organized retrieval**
- (arXiv 2025) **Selection: Multi-level retrieval for long context**
- (arXiv 2024) **LongRAG: A Dual-Stage Retrieval Framework for Long Context**
- (arXiv 2024) **From local to global: A graph rag approach to query-focused summarization** [[Paper]](https://arxiv.org/abs/2404.16130)

#### Graph / KG / Community Structures
- (*SEM 2025) **HSGM: Hierarchical segment-graph memory for scalable long-text semantics** [[Paper]](https://aclanthology.org/2025.starsem-1.28/)
- (EMNLP-IJCNLP 2019) **PullNet: Open domain question answering with iterative retrieval on knowledge bases and text**
- (arXiv 2024) **From local to global: A graph rag approach to query-focused summarization** [[Paper]](https://arxiv.org/abs/2404.16130)
- (arXiv 2022) **KG-o1: Enhancing Multi-hop Question Answering in LLMs with Knowledge Graphs**
- (EMNLP 2022) **UniKGQA: Unified Knowledge Graph Question Answering**
- (arXiv 2025) **KG-o1: Enhancing Multi-hop Question Answering in LLMs with Knowledge Graphs**

#### Long-Context Evidence
- (arXiv 2024) **LongRAG: A Dual-Stage Retrieval Framework for Long Context**
- (NAACL 2025) **LOFT: Long-context fine-tuning for retrieval augmentation**
- (arXiv 2025) **Can memory-augmented language models generalize on reasoning-in-a-haystack tasks?**
- (NAACL 2022) **LongT5: Efficient text-to-text transformer for long sequences**
- (arXiv 2024) **Long context vs. RAG for LLMs: An evaluation and revisits** [[Paper]](https://arxiv.org/abs/2501.01880)

---

### C. Next-Step Execute Plan

#### Rule-based control
- (Findings 2023) **Measuring and Narrowing the Compositionality Gap in Language Models**
- (ICLR 2022) **React: Synergizing reasoning and acting in language models**
- (ACL 2023) **Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions**
- (NAACL 2025) **Resource-Friendly Dynamic Enhancement Chain for Multi-hop QA (DEC)** [[Paper]](https://aclanthology.org/2025.naacl-long.650/)

#### Policy-based control
- (ACL 2024) **End-to-End Beam Retrieval for Multi-Hop Question Answering** [[Paper]](https://aclanthology.org/2024.acl-long.315/)
- (arXiv 2024) **RAGAR: Your falsehood radar: Rag-augmented reasoning for political fact-checking using multimodal large language models**
- (arXiv 2025) **Stop-RAG: Value-Based Retrieval Control for Iterative RAG** [[Paper]](https://arxiv.org/abs/2510.14337)
- (NAACL 2025) **SIM-RAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains** [[Paper]](https://aclanthology.org/2025.naacl-long.575/)

#### Search-based control
- (NeurIPS 2023) **Tree of thoughts: Deliberate problem solving with large language models**
- (arXiv 2024) **MindStar: Enhancing reasoning with large language models using search**
- (arXiv 2024) **Monte Carlo tree search for reasoning with language models**
- (ACL 2024) **End-to-End Beam Retrieval for Multi-Hop Question Answering** [[Paper]](https://aclanthology.org/2024.acl-long.315/)
- (EMNLP 2022) **UniKGQA: Unified Knowledge Graph Question Answering**

#### Verifier / PRM-score triggers
- (ICLR 2023) **Let’s verify step by step**
- (EMNLP 2023) **SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models**
- (Intelligent Computing 2024) **TaskMatrix.AI: Completing tasks by connecting foundation models with millions of APIs**
- (NAACL 2025) **SIM-RAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains** [[Paper]](https://aclanthology.org/2025.naacl-long.575/)
- (arXiv 2025) **Credible plan-driven RAG method for multi-hop question answering (PAR RAG)**

#### Planner--executor triggers
- (ICLR 2022) **Decomposed prompting: A modular approach for solving complex tasks**
- (arXiv 2025) **PECAN: Planning and executing code agents**
- (SIGIR 2025) **Rearter: Retrieval-augmented reasoning with trustworthy process rewarding**
- (ACL 2024) **WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models** [[Paper]](https://aclanthology.org/2024.acl-long.371/)
- (Journal of Technology Innovation and Engineering 2025) **Lightweight network-based semantic segmentation for uavs and its risc-v implementation**
- (Computer Simulation in Application 2025) **BiDeepLab: An improved lightweight multi-scale feature fusion DeepLab algorithm for facial recognition on mobile devices**

#### Uncertainty / confidence triggers
- (arXiv 2024) **Uncertainty-of-thought: Uncertainty estimation for chain-of-thought reasoning**
- (arXiv 2025) **Stop overthinking: A survey on efficient reasoning for large language models**
- (arXiv 2025) **Knowing when to stop: Dynamic context cutoff**
- (NAACL 2025) **SIM-RAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains** [[Paper]](https://aclanthology.org/2025.naacl-long.575/)
- (arXiv 2025) **Stop-RAG: Value-Based Retrieval Control for Iterative RAG** [[Paper]](https://arxiv.org/abs/2510.14337)

---

### D. Stop / Continue Criteria

#### Resource-constrained stopping
- (Findings 2023) **Measuring and Narrowing the Compositionality Gap in Language Models**
- (ICLR 2022) **React: Synergizing reasoning and acting in language models**
- (ACL 2023) **Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions**
- (arXiv 2024) **Dragin: dynamic retrieval augmented generation based on the information needs of large language models**
- (arXiv 2024) **LongRAG: Leveraging Long Context to Enhance Retrieval-Augmented Generation**
- (NAACL 2025) **Resource-Friendly Dynamic Enhancement Chain for Multi-hop QA (DEC)** [[Paper]](https://aclanthology.org/2025.naacl-long.650/)
- (arXiv 2025) **Your models have thought enough: Training large reasoning models to stop overthinking**
- (NeurIPS 2024) **TaskBench: Benchmarking large language models for task automation**

#### Confidence- / uncertainty-based stopping
- (arXiv 2025) **Stop overthinking: A survey on efficient reasoning for large language models**
- (arXiv 2025) **From system 1 to system 2: A survey of reasoning large language models**
- (IEEE 2025) **Decoding student cognitive abilities: a comparative study of explainable ai algorithms in educational data mining**
- (arXiv 2024) **Uncertainty-of-thought: Uncertainty estimation for chain-of-thought reasoning**
- (arXiv 2025) **Knowing when to stop: Dynamic context cutoff**
- (IEEE 2025) **Energy-constrained motion planning and scheduling for autonomous robots in complex environments**
- (IEEE 2025) **Reinforcement learning approach for highway lane-changing: PPO based strategy design**

#### Verifier- / prompt-based stopping
- (ACL 2023) **RARR: Researching and revising what language models say, using language models**
- (EMNLP 2023) **SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models**
- (Findings 2024) **Chain-of-verification reduces hallucination in large language models** [[Paper]](https://aclanthology.org/2024.findings-acl.209/)
- (NAACL 2025) **SIM-RAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains** [[Paper]](https://aclanthology.org/2025.naacl-long.575/)
- (arXiv 2025) **Stop-RAG: Value-Based Retrieval Control for Iterative RAG** [[Paper]](https://arxiv.org/abs/2510.14337)
- (NAACL 2025) **Resource-Friendly Dynamic Enhancement Chain for Multi-hop QA (DEC)** [[Paper]](https://aclanthology.org/2025.naacl-long.650/)
- (SIGIR 2025) **Rearter: Retrieval-augmented reasoning with trustworthy process rewarding**
- (ACL 2023) **Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions**
- (Findings 2023) **Measuring and Narrowing the Compositionality Gap in Language Models**
- (arXiv 2025) **Credible plan-driven RAG method for multi-hop question answering (PAR RAG)**
- (ACL 2024) **End-to-End Beam Retrieval for Multi-Hop Question Answering** [[Paper]](https://aclanthology.org/2024.acl-long.315/)
- (ICLR 2024) **Raptor: Recursive abstractive processing for tree-organized retrieval**
- (arXiv 2024) **LongRAG: Leveraging Long Context to Enhance Retrieval-Augmented Generation**
- (arXiv 2024) **LongRAG: A Dual-Stage Retrieval Framework for Long Context**
- (EMNLP 2022) **Knowledge boundary of large language models: A survey**
- (NAACL 2025) **SIM-RAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains** [[Paper]](https://aclanthology.org/2025.naacl-long.575/)
- (arXiv 2025) **Stop-RAG: Value-Based Retrieval Control for Iterative RAG** [[Paper]](https://arxiv.org/abs/2510.14337)

---

## Datasets & Benchmarks
- (EMNLP 2018) **HotpotQA: A dataset for diverse, explainable multi-hop question answering**
- (EMNLP 2020) **Constructing a dataset for multi-hop question answering**
- (NeurIPS 2024) **MEQA: A benchmark for multi-hop event-centric question answering with explanations**
- (arXiv 2024) **FanOutQA**
- (arXiv 2025) **Does your chain-of-thought prompt have a hub?**

## Surveys & Meta
- (FnTIR 2024) **Multi-hop question answering**
- (arXiv 2025) **Decoding student cognitive abilities: a comparative study of explainable ai algorithms in educational data mining**
- (ACL 2025) **Knowledge boundary of large language models: A survey** [[Paper]](https://aclanthology.org/2025.acl-long.251/)
- (arXiv 2023) **Retrieval-augmented generation for large language models: A survey**

## Contributing
- PRs are welcome! If you add a paper, please also add a **case label** consistent with the taxonomy in the survey.
- If you know an official code repo, add a `[[Code]](URL)` link after the `[[Paper]]` link.
- (Optional) If `Code` is a GitHub repo, you can append a star badge:
  - `[![Stars](https://img.shields.io/github/stars/ORG/REPO?style=social)](https://github.com/ORG/REPO)`

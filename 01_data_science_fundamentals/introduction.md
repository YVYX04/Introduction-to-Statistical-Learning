# The Advent of Data Science & Statistical Learning

## 1. Introduction

Over the past decade, data science and artificial intelligence (AI) have moved from specialized research domains to central pillars of economic, scientific, and political debate. Breakthroughs in large-scale machine learning (AlexNet), the rapid diffusion of cloud computing (AWS, Microsoft Azure, Oracle), and the public release of powerful generative models (ChatGPT 3.5. by OpenAI in November 2022) have profoundly altered how information is produced, processed, and consumed. Milestones such as the deployment of large language models (LLMs) to the general public, major advances in protein-structure prediction (Demis Hassabis's award of the 2024 Nobel Prize with *AlphaFold* from Google Deepmind), and the widespread adoption of AI systems in finance, healthcare, and creative industries have reinforced the perception that a structural technological shift is underway.

This acceleration is reflected in this year’s Time Magazine cover, which designates “The Architects of AI” as the symbolic figures of the moment. Beyond individual personalities, the cover captures a broader reality: AI is no longer confined to laboratories but embedded in large technology firms, startups, public institutions, and everyday tools. Investment volumes, regulatory initiatives, and public discourse between 2020 and 2025 all point to AI becoming a general-purpose technology with far-reaching implications.

<figure style="text-align: center;">
  <img src="images/times_2025.jpg" alt="Times 2025 cover" style="width: 60%;">
  <figcaption><strong>Fig. 1: Time Magazine (2025): “The Architects of AI”</strong>. <small>From left to right: Mark Zuckerberg (Meta), Lisa Su (AMD), Elon Musk (xAI, TESLA, SpaceX), Jensen Huang (Nvidia), Sam Altman (OpenAI), Demis Hassabis (Google Deepmind), Dario Amodei (Anthropic), and Fei-Fei Li (AI researcher). </small></figcaption>
</figure>

Yet, despite its prominence, fundamental questions remain unresolved. What does data science precisely encompass as a discipline? What technical and conceptual foundations lie behind the term "artificial intelligence"? And to what extent do recent developments represent genuine novelty rather than the maturation of long-standing statistical and computational ideas? In the end, what is truly data science? What does AI hides behind its fancy hood? Are those really new?

This repository's purpose is to lift the hood, to explore what lies behind the technical jargon, and understand how to exploit those tools. Before delving into more technical aspects, I introduce broad definitions of data science and machine learning (ML) / statistical learning.

## 2. What is Data Science?

In his book *Data science from scratch*, Grus (2019, p. 2) defines a data scientist as "someone who extracts insights from messy data". Indeed, it is arduous to precisely define data science since it is an interdisciplinary aggregate of numerous disciplines, such as computer science, statistics, algorithms, and software engineering. Data science is simultaneously an academic discipline, a research paradigm, a research method, a workflow, and a profession (Mike & Hazzan, 2023). Within the scope of this repository, we will stick to the first definition provided by Grus.

## 3. What is Statistical Learning?

Statistical learning / machine learning (ML) is closely linked to data science and could even be considered as being a subset of it. In general, ML is confronted with estimating a function $f$, such as in:

$$
Y = f(X) + \varepsilon
$$

where $Y$ is defined as the *target variable*, *dependent variable*, or *response variable*. $X$, on the other hand, is usually defined as being a set of predictors where $X = \{X_1, X_2, \cdots, X_n\}$, and $\varepsilon$ in a random *error term* independent of $X$ and has $\mathbb{E}(\varepsilon) = 0$. In essence, machine learning is the science of approximating $f$ for either realizing predictions or for inference purposes (Hastie et al., 2009, p. 17). *Supervised learning* occurs when the target variable is known, i.e. the data in the training set are *labelled* (e.g. the `MNIST` data set). On the other hand, when we have unlabelled data, e.g. we perform a clustering task for customers classification, we speak of *unsupervised learning* (we lack the response variable to supervise the "real performance" of our models). Further notions will be discussed in this repository.

## 4. Purpose of this Repository

The purpose of this repository is to blend conceptual knowledge with technical expertise in the fields of data science and machine learning. It is not destined to be an exhaustive overview of both disciplines, not in breadth, neither in depth, but rather a collection of selected and fundamentals topics. In the end, the objective is to build some small-scale data mining / ML projects for practical applications. Finally, the reader should not that the core reference of this entire repository will be Hastie et al.'s landmark book *An Introduction to Statistical Learning*. 


## References

- Grus, J. (2019). *Data science from scratch: first principles with python*. O'Reilly Media.

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *An introduction to statistical learning*.

- Mike, K., & Hazzan, O. (2023). What is data science?. *Communications of the ACM, 66(2)*, 12-13.
# Natural language processing course 2023/24: `Slovenian Instruction-based Corpus Generation`

#### Group members
- Gašper Spagnolo,
- Žiga Klun, 
- Žiga Črv

## Project description
This project explores the utilization of Large Language Models (LLMs) for conversational agent development in the Slovene language. We investigate various state-of-the-art LLMs suitable for fine-tuning, considering factors such as compatibility with Slovene, computational infrastructure requirements, and model capabilities. Emphasis is placed on understanding the creation process of LLMs and the construction of high-quality conversational datasets.
Our methodology involves reviewing datasets and categorizing instructions for training Instruce-based LLMs. We devise a comprehensive plan for data gathering, identifying sources such as med-over.net and slo-tech forums. Crawlers are developed to efficiently collect conversational data, which is organized systematically to facilitate fine-tuning of LLMs.
Additionally, we examine pertinent literature, including research on models like BLOOM and LLaMa 2, to ascertain key considerations in data preparation. By synthesizing these insights, we prepare a corpus of conversational data tailored for fine-tuning LLMs, ensuring its relevance and quality.
Furthermore, we discuss the potential adaptation of an existing LLM using the gathered data, offering insights into the practical application of our methodology. Our findings are consolidated in a final report, providing a comprehensive overview of the process and its implications for developing conversational agents in Slovene using LLMs.

### Proposed metodology
1. Review usable LLMs, select one that you might use (e.g., within SLING infrastructure, VEGA, Nvidia A100 GPUs).
2. (main goal of the project) Review datasets construction and categorization of instructions for selected Instruce-based LLMs. Prepare a plan for data gathering and identify sources (e.g., med-over.net, slo-tech forum, ...). Write crawlers, ... organize data in a way that is useful for "fine-tuning" the model. Check papers (e.g., BLOOM's, LLaMa 2's) to get to know, what aspects are important when preparing data.
3. Use the data to adapt an existing model using your data (optional).
4. Report on all your findings in the final report

### References
- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe, Training language models to follow instructions with human feedback, https://arxiv.org/abs/2203.02155.
- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. https://arxiv.org/abs/2307.09288.
- Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, ...  et al. (300+ additional authors not shown), BLOOM: A 176B-Parameter Open-Access Multilingual Language Model, https://arxiv.org/abs/2211.05100, model: https://huggingface.co/bigscience/bloom 


## Course obligations
### Submission 1

Project selection & simple corpus analysis:

- [x] Group (three members) selection
- [x] Report containing Introduction, existing solutions/related work and initial ideas
- [x] Well organized repository

### Submission 2

Initial implementation / baseline with results:

- [x] Updated Submission 1 parts
- [x] Implemented at least one solution with analysis
- [x] Future directions and ideas
- [x] Well organized repository

### Submission 3

Final solution and report:

- [ ] Final report incl. analyses and discussions
- [ ] Fully reproducible repository

### Peer review (PIVO)

Evaluate your peer group's work
- [ ] Each group will check final submissions of two other peer groups having the same topic


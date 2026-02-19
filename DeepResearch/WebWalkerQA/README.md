---
license: apache-2.0
task_categories:
- question-answering
language:
- zh
- en
---
📑 The paper of WebWalkerQA is available at [arXiv](https://arxiv.org/pdf/2501.07572).

📊 The dataset resource is a collection of **680** questions and answers from the WebWebWalker dataset.

🙋 The dataset is in the form of a JSON file.
The keys in the JSON include:
Question, Answer, Root_Url, and Info. The Info field contains
more detailed information, including Hop, Domain, Language,
Difficulty_Level, Source Website, and Golden_Path.
```
{
    "Question": "When is the paper submission deadline for the ACL 2025 Industry Track, and what is the venue address for the conference?",
    "Answer": "The paper submission deadline for the ACL 2025 Industry Track is March 21, 2025. The conference will be held in Brune-Kreisky-Platz 1.",
    "Root_Url": "https://2025.aclweb.org/",
    "Info":{
        "Hop": "multi-source",
        "Domain": "Conference",
        "Language": "English",
        "Difficulty_Level": "Medium",
        "Source_Website": ["https://2025.aclweb.org/calls/industry_track/","https://2025.aclweb.org/venue/"],
        "Golden_Path": ["root->call>student_research_workshop", "root->venue"]
    }
}
```
🏋️ We also release a collection of **15k** silver dataset, which although not yet carefully human-verified, can serve as supplementary \textbf{training data} to enhance agent performance.

🙋 If you have any questions, please feel free to contact us via the [Github issue](https://github.com/Alibaba-NLP/WebWalker/issue).

⚙️ Due to the web changes quickly, the dataset may contain outdated information, such as golden path or source website. We encourage you to contribute to the dataset by submitting a pull request to the WebWalkerQA or contacting us.

💡 If you find this dataset useful, please consider citing our paper:
```bigquery
@article{wu2025webwalker,
  title={Webwalker: Benchmarking llms in web traversal},
  author={Wu, Jialong and Yin, Wenbiao and Jiang, Yong and Wang, Zhenglin and Xi, Zekun and Fang, Runnan and Zhang, Linhai and He, Yulan and Zhou, Deyu and Xie, Pengjun and others},
  journal={arXiv preprint arXiv:2501.07572},
  year={2025}
}
```
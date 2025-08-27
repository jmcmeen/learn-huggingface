# learn-huggingface 🤗

A collection of Jupyter notebooks for learning Hugging Face Transformers, covering everything from basic model usage to advanced fine-tuning techniques.

## 📚 What You'll Learn

This repository provides hands-on experience with:

- **Model Loading & Usage**: Loading pre-trained models and tokenizers
- **Text Classification**: Sentiment analysis, topic classification, and custom text classification
- **Token Classification**: Named Entit
y Recognition (NER) and Part-of-Speech tagging
- **Text Generation**: GPT-style text generation and controlled generation
- **Question Answering**: Extractive and generative QA systems
- **Fine-tuning**: Custom model training on your own datasets
- **Model Optimization**: Quantization, pruning, and deployment techniques
- **Multi-modal Models**: Working with vision-language models

## 🚀 Getting Started

### Prerequisites

- Python 3.9 - 3.12
- Jupyter Notebook or JupyterLab
- Basic understanding of Python and machine learning concepts

### Installation

1. Clone this repository:
```bash
git clone https://github.com/jmcmeen/learn-huggingface.git
cd learn-huggingface
```

2. Create a virtual environment:
```bash
python -m venv hf-env
source hf-env/bin/activate  
# On Windows: hf-env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Quick Start

```bash
jupyter lab
```

Start with the `01_introduction.ipynb` notebook and work your way through the numbered sequence.

## 📓 Notebooks
- [Getting started with Hugging Face](01_introduction.ipynb)
- [Understanding tokenization](02_tokenizers.ipynb)
- [Loading and using pre-trained models](03_model_loading.ipynb)
- [Using Hugging Face pipelines for common tasks](04_pipelines.ipynb)
- [Building text classifiers](05_text_classification.ipynb)
- [NER and POS tagging](06_token_classification.ipynb)
- [Generating text with language models](07_text_generation.ipynb)
- [Building QA systems](08_question_answering.ipynb)
- [Text summarization techniques](09_summarization.ipynb)
- [Introduction to fine-tuning](10_fine_tuning_basics.ipynb)
- [Working with custom data](11_custom_datasets.ipynb)
- [Advanced training techniques](12_advanced_fine_tuning.ipynb)
- [Optimization and compression](13_model_optimization.ipynb)
- [Deploying models in production](14_deployment.ipynb)
- [Vision-language models](15_multimodal.ipynb)

## 🔗 Essential Resources

### Official Documentation
- [🤗 Hugging Face Documentation](https://huggingface.co/docs)
- [🤗 Transformers Library](https://huggingface.co/docs/transformers)
- [🤗 Datasets Library](https://huggingface.co/docs/datasets)
- [🤗 Tokenizers Library](https://huggingface.co/docs/tokenizers)
- [🤗 Hub Documentation](https://huggingface.co/docs/hub)

### PyTorch Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [torch.nn Documentation](https://pytorch.org/docs/stable/nn.html)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)

### Video Tutorials

#### Beginner-Friendly
- [Hugging Face Course - Complete Playlist](https://www.youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o) - Official Hugging Face course
- [Getting Started with Hugging Face in 15 Minutes](https://www.youtube.com/watch?v=QEaBAZQCtwE) - Quick overview
- [Fine-tuning BERT for Text Classification](https://www.youtube.com/watch?v=hinZO--TEk4) - Practical fine-tuning

#### Advanced Topics
- [Advanced Fine-tuning Techniques](https://www.youtube.com/watch?v=5T-iXNNiwIs) - Parameter efficient training
- [Deploying Hugging Face Models](https://www.youtube.com/watch?v=ND3JsDZqHyM) - Production deployment
- [Custom Tokenizers from Scratch](https://www.youtube.com/watch?v=MR8tBwXhowQ) - Building tokenizers



## 📊 Datasets

The notebooks use various datasets from the Hugging Face Hub:
- [IMDB Movie Reviews](https://huggingface.co/datasets/imdb) - Sentiment analysis
- [CoNLL-2003](https://huggingface.co/datasets/conll2003) - Named Entity Recognition
- [SQuAD](https://huggingface.co/datasets/squad) - Question Answering
- [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) - Summarization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Issues and Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/jmcmeen/learn-huggingface/issues) page
2. Search existing issues before creating a new one
3. Provide detailed information about your environment and the problem

## 📄 License

This project is licensed under the CC0 1.0 Universal - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face team](https://huggingface.co/huggingface) for creating amazing tools and resources
- The open-source community for continuous contributions
- All the researchers and developers who make their models available

## 🔗 Connect

- Follow [@huggingface](https://twitter.com/huggingface) on Twitter
- Join the [Hugging Face Discord](https://discord.com/invite/JfAtkvEtRb)
- Check out the [Hugging Face Forum](https://discuss.huggingface.co/)

---

**Happy Learning! 🚀**

*The best way to learn is by doing. Start with the basics and gradually work your way up to more complex topics.*
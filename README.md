# Chatlas - AI-Powered Sitemap Chat Extension

**Chatlas** is an intelligent Chrome extension that transforms any website's sitemap into an interactive AI chat interface. It automatically processes sitemap content, creates semantic embeddings for intelligent search, and enables users to have natural conversations about website documentation and content using advanced AI language models.

## What Chatlas Does

Chatlas bridges the gap between static website documentation and dynamic, conversational AI assistance. Instead of manually searching through extensive documentation or website content, users can simply:

1. **Process any website's sitemap** - Extract and analyze content from XML sitemaps
2. **Ask natural language questions** - Chat with the processed content using AI
3. **Get contextual answers** - Receive intelligent responses based on relevant website sections
4. **Save time and effort** - Skip manual browsing and searching through documentation

The extension uses cutting-edge AI technologies including semantic embeddings for content understanding and large language models for generating human-like responses, all packaged in a user-friendly Chrome extension interface.

## ✨ Features

- **🗺️ Sitemap Processing**: Automatically extracts and processes content from website sitemaps
- **🤖 AI-Powered Chat**: Ask questions about website content using advanced language models
- **🔍 Intelligent Search**: Uses embeddings to find the most relevant content for your queries
- **⏱️ Real-time Progress**: Track processing status with live updates
- **☁️ Cloud-Native**: Scalable backend architecture with AWS Fargate, DynamoDB, and S3
- **🛡️ Enterprise Security**: Protected with AWS WAF, rate limiting, and geographic blocking
- **📱 User-Friendly**: Clean, intuitive Chrome extension interface
- **⚡ High Performance**: Optimized for fast response times and efficient resource usage
- **🔄 Caching**: Intelligent caching system to avoid reprocessing the same sitemaps
- **🌍 Multi-Language Support**: Works with websites in multiple languages


### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Chrome Extension API, HTML5, CSS3, JavaScript | User interface and browser integration |
| **Backend** | Python Flask, Docker | API server and business logic |
| **Infrastructure** | AWS Fargate, ALB, WAF | Scalable container orchestration |
| **Database** | DynamoDB, S3 | Progress tracking and document storage |
| **AI/ML** | Hugging Face API, BGE-Large-EN, DeepSeek-R1 | Embeddings and chat completion |
| **Security** | AWS WAF, Rate Limiting, CORS | Protection against abuse and attacks |

## 🚀 Quick Start

### Prerequisites

- **Chrome Browser** (version 88+)
- **Hugging Face API Token** - [Get one here](https://huggingface.co/settings/tokens)
- **Docker** (optional, for local development)

### Installation

1. **Clone the Repository**

2. **Install Chrome Extension**
- Open Chrome and navigate to `chrome://extensions/`
- Enable **Developer Mode** (toggle in top-right corner)
- Click **Load unpacked**
- Select the `chatlas` folder (repository root)
- Pin the extension to your toolbar for easy access

3. **Get Started**
- Click the Chatlas extension icon in your Chrome toolbar
- Enter your Hugging Face API token
- Paste a sitemap URL and start processing!



## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow existing code style and formatting
- Add tests for new functionality
- Update documentation for API changes
- Test Chrome extension in multiple Chrome versions
- Ensure all security best practices are followed

### Areas for Contribution

- **Frontend Improvements**: Enhanced UI/UX, better error handling
- **Backend Optimization**: Performance improvements, additional AI models
- **Documentation**: Better examples, tutorials, API documentation
- **Testing**: Unit tests, integration tests, end-to-end testing
- **Security**: Additional security measures, vulnerability assessments

## 📈 Performance & Monitoring

### Production Monitoring

The production deployment includes comprehensive monitoring:

- **📊 CloudWatch Metrics**: API response times, error rates, request volumes
- **🛡️ WAF Logs**: Security events, blocked requests, geographic data
- **💿 DynamoDB Metrics**: Read/write capacity, throttling events
- **📱 Extension Analytics**: Usage patterns, error tracking
- **🔍 Application Logs**: Detailed debugging and audit trails

### Performance Optimization

- **Caching Strategy**: Intelligent caching to reduce processing time
- **Content Compression**: Gzip compression for all API responses
- **Database Optimization**: Efficient DynamoDB and S3 usage patterns
- **AI Model Selection**: Optimized models for speed and accuracy
- **Resource Management**: Efficient memory and CPU usage

## 🔄 Deployment

### Production Infrastructure

The production deployment runs on AWS with the following components:

- **ECS Fargate**: Auto-scaling container orchestration
- **Application Load Balancer**: High availability and SSL termination
- **AWS WAF**: Advanced threat protection and rate limiting
- **DynamoDB**: Fast, scalable NoSQL database for progress tracking
- **S3**: Secure, scalable object storage for processed content
- **CloudWatch**: Comprehensive logging and monitoring


## 🙏 Acknowledgments

- **Hugging Face** for providing excellent AI model APIs and infrastructure
- **AWS** for reliable, scalable cloud infrastructure services
- **Chrome Extensions Team** for the robust and well-documented extension platform
- **Open Source Community** for inspiration, libraries, and continuous support
- **Flask Community** for the lightweight and powerful web framework
- **Docker** for containerization technology that simplifies deployment

## 📞 Support & Contact

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/sc30ash/chatlas/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/sc30ash/chatlas/discussions)
- **📧 General Questions**: Create an issue with the `question` label
- **🔒 Security Issues**: Please report security vulnerabilities via private disclosure

### Community

- **Star this repository** if you find Chatlas useful
- **Share with others** who might benefit from AI-powered documentation chat
- **Contribute** to help make Chatlas even better
- **Follow updates** by watching the repository

---

<div align="center">

**Built with ❤️ using Flask, Chrome Extensions API, AWS, and Hugging Face AI**

[⭐ Star this repo](https://github.com/sc30ash/chatlas) • [🐛 Report Bug](https://github.com/sc30ash/chatlas/issues) • [💡 Request Feature](https://github.com/sc30ash/chatlas/discussions)

**Made by [sc30ash](https://github.com/sc30ash) | © 2025 | MIT License**

</div>





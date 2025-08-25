# Privacy Policy for Chatlas Chrome Extension

**Effective Date:** August 25, 2025  
**Last Updated:** August 25, 2025

## Overview

Chatlas is a Chrome extension that transforms website sitemaps into interactive AI-powered chat interfaces. This privacy policy explains how we collect, use, and protect your information when you use our extension.

**We are committed to protecting your privacy. Chatlas is designed with privacy-by-design principles and operates with minimal data collection.**

## Information We Collect

### Data Collected Locally (Stored on Your Device Only)

**Configuration Data:**
- Hugging Face API token (encrypted and stored locally)
- Previously processed sitemap URLs
- User interface preferences
- Chat history for current session

**Technical Data:**
- Extension usage statistics (locally stored)
- Error logs for troubleshooting (locally stored)
- Processing progress status

**This data is stored using Chrome's Storage API and never leaves your device except when explicitly required for functionality.**

### Data Processed Temporarily

**Sitemap Content:**
- XML sitemap files from websites you choose to process
- Extracted text content from webpage URLs found in sitemaps
- Generated embeddings from processed content

**Chat Data:**
- Questions you ask the AI assistant
- AI responses generated from your queries
- Context information used for generating responses

**Important:** This data is processed temporarily on our secure servers and is not stored permanently or used for any purpose other than providing the requested functionality.

## How We Use Your Information

### Primary Functionality
- **Sitemap Processing:** Extract and analyze content from XML sitemaps you provide
- **AI Chat Interface:** Generate intelligent responses to your questions using processed content
- **Progress Tracking:** Monitor and display processing status
- **Caching:** Temporarily store processed results to improve performance and reduce processing time

### Technical Operations
- **Error Handling:** Diagnose and resolve technical issues
- **Performance Optimization:** Improve response times and reliability
- **Security Monitoring:** Detect and prevent abuse of our services

## Data Sharing and Third Parties

### Third-Party Services We Use

**Hugging Face API:**
- Purpose: Generate embeddings and AI chat responses
- Data Shared: Text content from processed sitemaps, your questions
- Privacy Policy: [Hugging Face Privacy Policy](https://huggingface.co/privacy)
- Your API token is used for authentication

**AWS Infrastructure:**
- Purpose: Host our backend processing services
- Data Shared: Temporarily processed sitemap content, progress information
- Security: All data encrypted in transit and at rest
- Retention: Processed data cached for 24 hours, then automatically deleted

### We Do NOT:
- ❌ Sell your personal information to anyone
- ❌ Share your data with advertisers
- ❌ Store your Hugging Face API tokens on our servers
- ❌ Collect browsing history beyond the sitemaps you choose to process
- ❌ Track your activity across other websites
- ❌ Use your data for marketing purposes

## Data Security

### Security Measures
- **Encryption:** All data transmitted using HTTPS/TLS encryption
- **Local Storage:** Sensitive data like API tokens stored locally on your device
- **Access Controls:** Strict access controls on our backend infrastructure
- **AWS WAF Protection:** Advanced threat protection and rate limiting
- **Regular Security Audits:** Ongoing security assessments and improvements

### Data Retention
- **Local Data:** Stored indefinitely on your device until you remove the extension
- **Processed Content:** Cached on our servers for 24 hours, then automatically deleted
- **Chat Sessions:** Stored locally during your session, cleared when you close the extension

## Your Privacy Rights

### You Can:
- **Access:** View what data is stored locally by inspecting Chrome extension storage
- **Delete:** Remove all local data by uninstalling the extension
- **Control:** Choose which sitemaps to process and what questions to ask
- **Modify:** Update your API tokens and preferences at any time

### To Exercise Your Rights:
- **Delete Local Data:** Uninstall the extension from Chrome
- **Clear Chat History:** Use the clear history button in the extension
- **Remove Cached Data:** Contact us to request removal of cached processing results

## Permissions Justification

### Chrome Extension Permissions

**"storage"**
- **Purpose:** Store your API tokens and preferences locally on your device
- **Data:** Configuration settings, API tokens (encrypted), chat history

**"tabs"**
- **Purpose:** Detect current website to suggest sitemap URLs
- **Data:** Current tab URL to help identify potential sitemap locations

**"webRequest" and "webRequestBlocking"**
- **Purpose:** Fetch and process XML sitemap files from websites
- **Data:** HTTP requests to sitemap URLs you specify

**"sidePanel"**
- **Purpose:** Provide the main user interface for the extension
- **Data:** No additional data collected through this permission

**"activeTab"**
- **Purpose:** Access current tab URL to suggest relevant sitemaps
- **Data:** Current webpage URL only when extension is actively used

**"host_permissions": ["*://*/*"]**
- **Purpose:** Access sitemaps from any website you choose to process
- **Data:** Only sitemap files from URLs you explicitly provide

## Children's Privacy

Chatlas is not intended for use by children under 13. We do not knowingly collect personal information from children under 13. If you believe a child has provided personal information to us, please contact us immediately.

## Changes to This Privacy Policy

We may update this privacy policy from time to time. When we do:
- We will update the "Last Updated" date at the top of this policy
- Significant changes will be announced through the extension interface
- You will be asked to review and accept updated policies before continued use

## International Users

Chatlas is operated from the United States. If you are using our extension from outside the US, please be aware that your information may be transferred to, stored, and processed in the United States where our servers are located.

## Data Minimization

We follow data minimization principles:
- **Collect Only What's Needed:** We only process data necessary for core functionality
- **Temporary Processing:** Most data is processed temporarily and not stored long-term
- **Local Storage Preference:** Sensitive data like API tokens stays on your device
- **Automatic Deletion:** Cached data is automatically deleted after 24 hours

## Contact Information

If you have questions about this privacy policy or our data practices:

- **GitHub Issues:** [https://github.com/sc30ash/chatlas/issues](https://github.com/sc30ash/chatlas/issues)
- **Email:** Create an issue on GitHub with your privacy-related questions
- **Developer:** sc30ash

## Compliance

This privacy policy is designed to comply with:
- Chrome Web Store Developer Program Policies
- California Consumer Privacy Act (CCPA)
- General Data Protection Regulation (GDPR)
- Children's Online Privacy Protection Act (COPPA)

## Open Source Transparency

Chatlas is open source software. You can:
- **Review Code:** Examine our source code at [https://github.com/sc30ash/chatlas](https://github.com/sc30ash/chatlas)
- **Verify Claims:** Confirm our privacy practices by reviewing the code
- **Contribute:** Help improve privacy and security through contributions
- **Audit:** Conduct your own security and privacy audits

---

**By using Chatlas, you agree to this privacy policy. If you do not agree with this policy, please do not use the extension.**

**Your privacy is important to us. If you have any concerns or questions, please don't hesitate to reach out.**

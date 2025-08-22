document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const setupView = document.getElementById('setup-view');
  const progressView = document.getElementById('progress-view');
  const chatView = document.getElementById('chat-view');
  const statusEl = document.getElementById('status');
  const messagesEl = document.getElementById('messages');
  const progressBar = document.getElementById('progress-bar');
  const progressMsg = document.getElementById('progress-message');
  
  // Add event listener for dropdown change (add this in DOMContentLoaded)
const sitemapSelect = document.getElementById('sitemap_select');
const customUrlGroup = document.getElementById('custom_url_group');
const customUrlInput = document.getElementById('custom_sitemap_url');

// Add this after the existing element definitions
if (sitemapSelect) {
    sitemapSelect.addEventListener('change', (e) => {
        if (e.target.value === 'custom') {
            customUrlGroup.style.display = 'block';
            sitemapUrl = customUrlInput.value || '';
        } else {
            customUrlGroup.style.display = 'none';
            sitemapUrl = e.target.value;
        }
        
        // Update status
        if (sitemapUrl) {
            statusEl.textContent = 'Sitemap configured';
        }
        
        saveSettings();
    });
}

if (customUrlInput) {
    customUrlInput.addEventListener('input', (e) => {
        if (sitemapSelect.value === 'custom') {
            sitemapUrl = e.target.value;
            
            // Update status
            if (sitemapUrl) {
                statusEl.textContent = 'Custom sitemap configured';
            }
            
            saveSettings();
        }
    });
}


  // Buttons
  const setupBtn = document.getElementById('setup-btn');
  const configBtn = document.getElementById('config-btn');
  
  // Forms
  const chatForm = document.getElementById('chat-form');
  
  // State
  let currentTaskId = null;
  let backendUrl = 'http://sitemap-alb-40407797.us-east-2.elb.amazonaws.com';
  let hfToken = '';
  let sitemapUrl = '';
  
  // Initialize
  loadSettings();
  
  // Event Listeners
  setupBtn.addEventListener('click', startProcessing);
  configBtn.addEventListener('click', showSetupView);
  chatForm.addEventListener('submit', handleChatSubmit);
  
 function loadSettings() {
    chrome.storage.sync.get(['hfToken', 'sitemapUrl'], (data) => {
        if (data.hfToken) {
            hfToken = data.hfToken;
            document.getElementById('hf_token').value = data.hfToken;
        }
        
        // Load available sitemaps from current website
        loadAvailableSitemaps().then(() => {
            // If we had a saved sitemap URL, try to select it
            if (data.sitemapUrl) {
                const selectEl = document.getElementById('sitemap_select');
                const customUrlGroup = document.getElementById('custom_url_group');
                const customUrlInput = document.getElementById('custom_sitemap_url');
                const options = Array.from(selectEl.options);
                const foundOption = options.find(opt => opt.value === data.sitemapUrl);
                
                if (foundOption) {
                    selectEl.value = data.sitemapUrl;
                    sitemapUrl = data.sitemapUrl;
                    customUrlGroup.style.display = 'none';
                    statusEl.textContent = 'Sitemap configured';
                } else {
                    // Use custom option for saved URL not in robots.txt
                    selectEl.value = 'custom';
                    customUrlInput.value = data.sitemapUrl;
                    sitemapUrl = data.sitemapUrl;
                    customUrlGroup.style.display = 'block';
                    statusEl.textContent = 'Custom sitemap configured';
                }
            }
        });
    });
}


  
  function saveSettings() {
    chrome.storage.sync.set({ 
        hfToken, 
        sitemapUrl: sitemapUrl 
    });
   }

  
  function showMessage(message, type = 'info') {
    messagesEl.innerHTML = `<div class="${type}">${message}</div>`;
    setTimeout(() => messagesEl.innerHTML = '', 3000);
  }
  
  function showSetupView() {
    setupView.style.display = 'block';
    progressView.style.display = 'none';
    chatView.style.display = 'none';
  }
  
  function showProgressView() {
    setupView.style.display = 'none';
    progressView.style.display = 'block';
    chatView.style.display = 'none';
  }
  
  function showChatView() {
    setupView.style.display = 'none';
    progressView.style.display = 'none';
    chatView.style.display = 'block';
  }
  
  // Update the startProcessing function to use the dropdown value
function startProcessing() {
    // Get values from form elements
    hfToken = document.getElementById('hf_token').value;
    
    // Get sitemap URL from dropdown or custom input
    const sitemapSelect = document.getElementById('sitemap_select');
    const customUrlInput = document.getElementById('custom_sitemap_url');
    
    if (sitemapSelect && sitemapSelect.value === 'custom') {
        sitemapUrl = customUrlInput ? customUrlInput.value : '';
    } else if (sitemapSelect) {
        sitemapUrl = sitemapSelect.value;
    }
    
    // Debug log to verify
    console.log('Starting processing with:', { hfToken: !!hfToken, sitemapUrl });
    
    if (!backendUrl || !hfToken || !sitemapUrl) {
        showMessage('Please fill all fields', 'error');
        return;
    }
    
    // Save settings
    saveSettings();
    
    // Start processing
    fetch(`${backendUrl}/api/process`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            hf_token: hfToken,
            sitemap_url: sitemapUrl
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showMessage(data.error, 'error');
        } else {
            currentTaskId = data.task_id;
            showProgressView();
            checkProgress();
        }
    })
    .catch(error => {
        showMessage(`Failed to start processing: ${error.message}`, 'error');
    });
}


  
  function checkProgress() {
    if (!currentTaskId) return;
    
    fetch(`${backendUrl}/api/progress/${currentTaskId}`)
      .then(response => response.json())
      .then(data => {
        progressBar.style.width = `${data.progress}%`;
        progressMsg.textContent = data.message;
        
        if (data.complete) {
          if (data.error) {
            showMessage(`Error: ${data.error}`, 'error');
            showSetupView();
          } else {
            statusEl.textContent = 'Sitemap loaded';
            showChatView();
          }
        } else {
          setTimeout(checkProgress, 1000);
        }
      })
      .catch(error => {
        showMessage(`Progress check failed: ${error.message}`, 'error');
        showSetupView();
      });
  }
  
  function handleChatSubmit(e) {
    e.preventDefault();
    const question = document.getElementById('question').value.trim();
    
    if (!question) return;
    
    const responseContainer = document.getElementById('response-container');
    
    // Add user message
    responseContainer.innerHTML += `
        <div class="chat-message user-message">
            <div class="message-content">
                <strong>You:</strong> ${question}
            </div>
        </div>
    `;
    
    // Add loading message
    responseContainer.innerHTML += `
        <div class="chat-message bot-message loading" id="loading-msg">
            <div class="message-content">
                <strong>Assistant:</strong> <span class="thinking">Thinking...</span>
            </div>
        </div>
    `;
    
    // Clear input
    document.getElementById('question').value = '';
    
    fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            hf_token: hfToken,
            sitemap_url: sitemapUrl,
            question: question
        })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading message
        document.getElementById('loading-msg').remove();
        
        if (data.error) {
            responseContainer.innerHTML += `
                <div class="chat-message error-message">
                    <div class="message-content">
                        <strong>Error:</strong> ${data.error}
                    </div>
                </div>
            `;
        } else {
            // Format the response with markdown conversion
            const formattedResponse = formatMarkdownToHtml(data.response);
            responseContainer.innerHTML += `
                <div class="chat-message bot-message">
                    <div class="message-content">
                        <strong>Assistant:</strong>
                        <div class="response-text">${formattedResponse}</div>
                    </div>
                </div>
            `;
        }
        
        // Scroll to bottom
        responseContainer.scrollTop = responseContainer.scrollHeight;
    })
    .catch(error => {
        document.getElementById('loading-msg').remove();
        responseContainer.innerHTML += `
            <div class="chat-message error-message">
                <div class="message-content">
                    <strong>Error:</strong> ${error.message}
                </div>
            </div>
        `;
    });
}

});

// Add these new functions after the existing functions:

async function getCurrentTabDomain() {
    try {
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        if (tab && tab.url) {
            const url = new URL(tab.url);
            return url.protocol + '//' + url.hostname;
        }
    } catch (error) {
        console.error('Error getting current tab:', error);
    }
    return null;
}

async function fetchRobotsTxt(domain) {
    try {
        const robotsUrl = domain + '/robots.txt';
        const response = await fetch(robotsUrl);
        if (response.ok) {
            return await response.text();
        }
    } catch (error) {
        console.error('Error fetching robots.txt:', error);
    }
    return null;
}

function parseSitemapsFromRobots(robotsText) {
    if (!robotsText) return [];
    
    const sitemaps = [];
    const lines = robotsText.split('\n');
    
    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.toLowerCase().startsWith('sitemap:')) {
            const sitemapUrl = trimmed.substring(8).trim(); // Remove "Sitemap:" prefix
            if (sitemapUrl && (sitemapUrl.includes('sitemap') || sitemapUrl.endsWith('.xml'))) {
                sitemaps.push(sitemapUrl);
            }
        }
    }
    
    return [...new Set(sitemaps)]; // Remove duplicates
}

async function loadAvailableSitemaps() {
    const selectEl = document.getElementById('sitemap_select');
    const customGroupEl = document.getElementById('custom_url_group');
    
    // Show loading state
    selectEl.innerHTML = '<option value="">Loading sitemaps...</option>';
    
    // Get current tab domain
    const domain = await getCurrentTabDomain();
    if (!domain) {
        selectEl.innerHTML = '<option value="custom">Custom URL</option>';
        customGroupEl.style.display = 'block';
        return;
    }
    
    // Fetch and parse robots.txt
    const robotsText = await fetchRobotsTxt(domain);
    const sitemaps = parseSitemapsFromRobots(robotsText);
    
    // Clear loading state
    selectEl.innerHTML = '';
    
    // Add found sitemaps
    if (sitemaps.length > 0) {
        sitemaps.forEach(sitemap => {
            const option = document.createElement('option');
            option.value = sitemap;
            option.textContent = sitemap;
            selectEl.appendChild(option);
        });
        
        // Set first sitemap as selected and update sitemapUrl
        sitemapUrl = sitemaps[0];
        selectEl.value = sitemaps;
    }
    
    // Always add custom option
    const customOption = document.createElement('option');
    customOption.value = 'custom';
    customOption.textContent = '🔧 Custom URL';
    selectEl.appendChild(customOption);
    
    // Show custom input if no sitemaps found
    if (sitemaps.length === 0) {
        selectEl.value = 'custom';
        customGroupEl.style.display = 'block';
    }
}

// Replace the existing dropdown event listener with this updated version:
sitemapSelect.addEventListener('change', (e) => {
    const customUrlGroup = document.getElementById('custom_url_group');
    const customUrlInput = document.getElementById('custom_sitemap_url');
    
    if (e.target.value === 'custom') {
        customUrlGroup.style.display = 'block';
        sitemapUrl = customUrlInput.value || '';
    } else {
        customUrlGroup.style.display = 'none';
        sitemapUrl = e.target.value;
    }
    
    // Update status
    if (sitemapUrl) {
        statusEl.textContent = 'Sitemap configured';
    }
    
    saveSettings();
});

customUrlInput.addEventListener('input', (e) => {
    const sitemapSelect = document.getElementById('sitemap_select');
    if (sitemapSelect.value === 'custom') {
        sitemapUrl = e.target.value;
        
        // Update status
        if (sitemapUrl) {
            statusEl.textContent = 'Custom sitemap configured';
        }
        
        saveSettings();
    }
});

function formatMarkdownToHtml(text) {
    // Convert markdown formatting to HTML
    return text
        // Bold text **text** -> <strong>text</strong>
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Headers (if any)
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        // Single line breaks
        .replace(/\n/g, '<br>')
        // Wrap in paragraph if not already wrapped
        .replace(/^(?!<[h|p])/gm, '<p>')
        .replace(/(?<!>)$/gm, '</p>')
        // Clean up empty paragraphs
        .replace(/<p><\/p>/g, '')
        // List items - • or -
        .replace(/^- (.*$)/gim, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
        // Emojis and special characters remain as-is
}



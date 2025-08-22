// Handle side panel opening when extension icon is clicked
chrome.action.onClicked.addListener((tab) => {
  // Set options for this specific tab (remove await here)
  chrome.sidePanel.setOptions({
    tabId: tab.id,
    path: 'popup.html',
    enabled: true
  });
  
  // Open side panel only for this tab (keep synchronous)
  chrome.sidePanel.open({ tabId: tab.id });
});

// Optional: Clean up when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  try {
    chrome.sidePanel.setOptions({
      tabId: tabId,
      enabled: false
    });
  } catch (error) {
    // Tab already closed, ignore error
  }
});

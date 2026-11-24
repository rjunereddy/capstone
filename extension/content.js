// content.js - Extracts page payload for the Text/NLP Module

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ExtractText") {
        // Extract visible text from the DOM. We limit it to avoid massive payloads.
        let bodyText = document.body.innerText || "";
        bodyText = bodyText.substring(0, 2000); // Send first 2000 chars for DistilBERT
        
        sendResponse({ text: bodyText });
    }
    return true; 
});

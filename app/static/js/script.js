document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetBtn = document.getElementById('resetBtn');
    const videoFeed = document.getElementById('video-feed');
    const sentenceElement = document.getElementById('sentence');
    const statusElement = document.getElementById('status');
    
    // Variables
    let isRunning = false;
    let statusUpdateInterval = null;
    
    // Initialize buttons
    stopBtn.disabled = true;
    resetBtn.disabled = true;
    
    // Event listeners
    startBtn.addEventListener('click', startRecognition);
    stopBtn.addEventListener('click', stopRecognition);
    resetBtn.addEventListener('click', resetRecognition);
    
    // Functions
    function startRecognition() {
        if (isRunning) return;
        
        // Refresh video feed
        const timestamp = new Date().getTime();
        videoFeed.src = '/video_feed?' + timestamp;
        
        // Update UI
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        resetBtn.disabled = true;
        
        // Start status updates
        startStatusUpdates();
    }
    
    function stopRecognition() {
        if (!isRunning) return;
        
        // Update UI
        isRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = false;
        
        // Stop status updates
        stopStatusUpdates();
    }
    
    function resetRecognition() {
        // Reset video feed
        const timestamp = new Date().getTime();
        videoFeed.src = `/video_feed?${timestamp}`;
        
        // Update UI
        isRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = true;
        
        // Clear sentence
        sentenceElement.textContent = '';
        statusElement.textContent = 'Waiting for signs...';
        
        // Stop status updates
        stopStatusUpdates();
    }
    
    function startStatusUpdates() {
        // Clear any existing interval
        if (statusUpdateInterval) {
            clearInterval(statusUpdateInterval);
        }
        
        // Set up polling for status updates
        statusUpdateInterval = setInterval(() => {
            fetch('/get_status')
                .then(response => response.json())
                .then(data => {
                    // Update both the sentence and status displays
                    sentenceElement.textContent = data.status;
                    statusElement.textContent = data.status;
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }, 100); // Poll more frequently
    }
    
    function stopStatusUpdates() {
        if (statusUpdateInterval) {
            clearInterval(statusUpdateInterval);
            statusUpdateInterval = null;
        }
    }
    
    // Start status updates immediately when video loads
    videoFeed.addEventListener('load', () => {
        console.log('Video feed loaded');
        if (isRunning) {
            startStatusUpdates();
        }
    });
}); 
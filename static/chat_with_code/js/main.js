function validateForm() {
    var checkboxes = document.querySelectorAll('input[type="checkbox"][name="languages"]');
    var checked = false;
    checkboxes.forEach(function(checkbox) {
        if (checkbox.checked) {
            checked = true;
        }
    });
    if (!checked) {
        alert("Please select at least one language.");
        return false; // Prevent form submission
    }
    return true; // Allow form submission
}

document.addEventListener('DOMContentLoaded', function() {
    const splitter = document.getElementById('splitter');
    const rightPane = document.getElementById('right-pane');

    let isResizing = false;
    let initialMouseX;

    splitter.addEventListener('mousedown', function(e) {
        isResizing = true;
        initialMouseX = e.clientX;
        document.body.style.cursor = 'ew-resize';
    });

    document.addEventListener('mousemove', function(e) {
        if (!isResizing) return;
        const dx = initialMouseX - e.clientX;
        const newRightPaneWidth = rightPane.offsetWidth + dx;
        rightPane.style.width = `${newRightPaneWidth}px`;
        initialMouseX = e.clientX;
    });

    document.addEventListener('mouseup', function() {
        isResizing = false;
        document.body.style.cursor = 'default';
    });
});

function showLoading() {
    var loaderContainer = document.getElementById('loader-container');
    loaderContainer.style.display = 'block'; // Show loader container

    // Simulate loading delay (2 seconds) - replace with actual logic
    setTimeout(function() {
        // Normally, you would perform an asynchronous action (e.g., AJAX request) here

        // Simulate success after loading
        hideLoading(); // This function should be called after receiving a response from your backend
    }, 500000); // Adjust delay time as per your requirements
}

function hideLoading() {
    var loaderContainer = document.getElementById('loader-container');
    loaderContainer.style.display = 'none'; // Hide loader container
}

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.copy-button').forEach(button => {
        button.addEventListener('click', function() {
            const textareaId = this.getAttribute('data-code-block');
            const textarea = document.getElementById(textareaId);
            textarea.select();
            document.execCommand('copy');
            
            // Show the copied message next to the button
            const copiedMessage = this.querySelector('.copied-message');
            copiedMessage.classList.remove('hidden');

            // Hide the copied message after 2 seconds
            setTimeout(() => {
                copiedMessage.classList.add('hidden');
            }, 2000);
        });
    });
});

function toggleDrawer() {
    var drawer = document.getElementById('drawer');
    var overlay = document.getElementById('overlay');
    drawer.classList.toggle('open');
    overlay.classList.toggle('show');
}
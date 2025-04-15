document.addEventListener('DOMContentLoaded', () => {
    // Element selections
    const centerBlock = document.getElementById('centerBlock');
    const centerCaptions = document.getElementById('centerCaptions');
    const captionLine1 = document.getElementById('captionLine1');
    const captionLine2 = document.getElementById('captionLine2');
    const centerCaption = document.getElementById('centerCaption');
    const topLeftBlock = document.getElementById('topLeftBlock');
    const bottomLeftBlock = document.getElementById('bottomLeftBlock');
    const topRightBlock = document.getElementById('topRightBlock');
    const bottomRightBlock = document.getElementById('bottomRightBlock');
    const linesSvg = document.getElementById('linesSvg');
    const allLines = linesSvg.querySelectorAll('line');
    // NEW: Select the structure/sequence rows container
    const structureSequenceRows = document.getElementById('structureSequenceRows');

    const cornerBlocks = [topLeftBlock, bottomLeftBlock, topRightBlock, bottomRightBlock];

    const animationDuration = 5000; // Total loop time in ms (5 seconds)
    const lineFadeInDuration = 500; // Duration of line fade-in (matches CSS)
    const captionFadeDuration = 300; // Duration of caption fade (matches CSS)
    // NEW: Duration for structure/sequence fade-in (can match caption or be different)
    const structureSequenceFadeInDuration = 500; // Matches CSS transition

    console.log("Selected corner blocks:", cornerBlocks); // DEBUG
    console.log("Selected structure/sequence rows:", structureSequenceRows); // DEBUG

    function runAnimation() {
        console.log(`--- Animation Cycle Start (Time: ${Date.now()}) ---`); // DEBUG

        // --- Reset States (Start of Loop) ---
        console.log("Resetting corner blocks visibility and flames..."); // DEBUG
        cornerBlocks.forEach(block => {
            block.classList.remove('visible', 'flaming');
            void block.offsetWidth;
        });

        console.log("Resetting lines opacity..."); // DEBUG
        allLines.forEach(line => {
            line.style.transition = 'none';
            line.style.opacity = '0';
            void line.offsetWidth;
            line.style.transition = `opacity ${lineFadeInDuration / 1000}s ease-in-out`;
        });

        console.log("Resetting captions..."); // DEBUG
        captionLine1.innerHTML = '<span class="color-strong">Strong</span> Sequence Modeling';
        captionLine2.innerHTML = '<span class="color-limited">Limited</span> Structural Modeling';
        captionLine2.classList.remove('caption-fade-out', 'caption-fade-in');
        captionLine1.style.opacity = '1';
        captionLine2.style.opacity = '1';
        
        // Uncomment this to animate the variations from DPLM-2 to DPLM-2.1
        centerCaption.innerHTML = 'DPLM-2';
        centerCaption.classList.remove('caption-fade-out', 'caption-fade-in');
        centerCaption.style.opacity = '1';

        // NEW: Reset structure/sequence rows opacity
        // console.log("Resetting structure/sequence rows..."); // DEBUG
        // if (structureSequenceRows) {
        //     structureSequenceRows.style.transition = 'none'; // Disable transition for reset
        //     structureSequenceRows.style.opacity = '0';
        //      void structureSequenceRows.offsetWidth; // Force reflow
        //      // Re-enable transition for potential fade-in
        //      structureSequenceRows.style.transition = `opacity ${structureSequenceFadeInDuration / 1000}s ease-in-out`;
        // } else {
        //      console.error("Structure/Sequence Rows container not found!");
        // }


        // --- Animation Steps --- Schedule them

        // Step 1: Corner blocks fade in (after 1 second) + Flames
        setTimeout(() => {
            console.log("STEP 1: Adding .visible to corner blocks"); // DEBUG
            cornerBlocks.forEach(block => {
                if (block) {
                     block.classList.add('visible');
                     console.log(` -> Added 'visible' to ${block.id}`); // DEBUG
                } else {
                    console.error("Error: Corner block element not found during Step 1!"); // DEBUG
                }
                 // Add flames shortly after fade-in starts
                setTimeout(() => {
                    if(block) {
                        block.classList.add('flaming');
                        console.log(` -> Added 'flaming' to ${block.id}`); // DEBUG
                    }
                } , 150);
            });
        }, 1200); // 1 second delay

        // Step 2: Fade In Lines (after 2 seconds)
        setTimeout(() => {
            console.log("STEP 2: Setting lines opacity to 1"); // DEBUG
            allLines.forEach(line => {
                line.style.opacity = '1';
            });
        }, 2000); // 2 seconds delay

        // Step 3: Change bottom caption (AFTER lines finish fading in)
        const lineFadeEndTime = 2000 + lineFadeInDuration; // = 3000ms
        const captionChangeStartTime = lineFadeEndTime;
        const captionChangeEndTime = captionChangeStartTime + captionFadeDuration; // = 3300ms

        setTimeout(() => {
            console.log("STEP 3: Changing caption"); // DEBUG
            captionLine2.classList.add('caption-fade-out');
            
            // Uncomment this to animate the variations from DPLM-2 to DPLM-2.1
            centerCaption.classList.add('caption-fade-out');
            setTimeout(() => {
                captionLine2.innerHTML = '<span class="color-strong">Strong</span> Structural Modeling';
                captionLine2.classList.remove('caption-fade-out');
                captionLine2.classList.add('caption-fade-in');
                
                centerCaption.innerHTML = '<span class="color-2-5">DPLM-2.1</span>';
                centerCaption.classList.remove('caption-fade-out');
                centerCaption.classList.add('caption-fade-in');
            }, captionFadeDuration); // Inner timeout completes at captionChangeEndTime (3300ms)
        }, captionChangeStartTime); // Start caption change @ 3000ms

        // NEW Step 4: Fade in Structure/Sequence Rows (AFTER caption finishes changing)
        // Add a small delay for visual separation if needed (e.g., 100ms)
        const structureSequenceFadeInStartTime = captionChangeEndTime + 100; // Start @ 3400ms

        // setTimeout(() => {
        //     console.log("STEP 4: Fading in Structure/Sequence Rows"); // DEBUG
        //     if (structureSequenceRows) {
        //         structureSequenceRows.style.opacity = '1'; // Trigger fade-in via CSS transition
        //     }
        // }, structureSequenceFadeInStartTime); // Start fade-in @ 3400ms

    } // End of runAnimation

    // Run the animation immediately on load
    console.log("Initial animation run starting..."); // DEBUG
    runAnimation();

    // Set the animation to loop every 5 seconds
    console.log("Setting up animation loop interval..."); // DEBUG
    setInterval(runAnimation, animationDuration);
});
window.addEventListener('load', () => {
  function preloadImages() {
   const flows_imageSrc = (i) => `static/demo/flows/${String(i).padStart(5, '0')}.png`;
   const flows_maskSrc = (i) => `static/demo/flows/${String(i).padStart(5, '0')}_occ.png`
   const swing_imageSrc = (i) => `static/demo/swing/${String(i).padStart(5, '0')}.jpg`
   
   function preload(srcGetter) {
     result = [];
     for (var i = 0; i < 60; i++) {
       const img = new Image();
       img.src = srcGetter(i);
       result.push(img);
     }
     return result;
   }
   
   return {
     flow: preload(flows_imageSrc),
     mask: preload(flows_maskSrc),
     swing: preload(swing_imageSrc)
   }
  }
  images = preloadImages();

  // slide follow animation
  function displayFramesWithSlider(framePathPrefix, totalFrames) {
    const frame = document.getElementById("frame");
    const slider = document.getElementById("slider");
    const sliderLabel = document.getElementById("slider-label");

    // init slider
    slider.setAttribute("max", totalFrames);
    slider.setAttribute("min", 1);

    slider.addEventListener("input", function () {
      const frameIndex = slider.value;
      const framePath = framePathPrefix + String(frameIndex).padStart(5, '0') + ".jpg";
      frame.src = framePath;
      // sliderLabel.textContent = "Slider: " + frameIndex;
    });
  }

  // Usage
  displayFramesWithSlider("static/demo/swing/", 59); // Example with 100 frames

  const imageBoard1 = document.querySelectorAll('.image-board')[0];
  const clickableImage1 = imageBoard1.querySelector('.clickable-image');
  const imageBoard2 = document.querySelectorAll('.image-board')[1];
  const clickableImage2 = imageBoard2.querySelector('.clickable-image');
  const initialDot = document.getElementById('initial-dot');
  const slider = document.getElementById('slider');
  const clearBtn = document.getElementById('clear-btn');
  clearBtn.addEventListener('click', function () {
    // remove original dot & add a fixed dot
    const existingDots1 = imageBoard1.querySelectorAll('.dot');
    existingDots1.forEach(dot => imageBoard1.removeChild(dot));
    const existingDots2 = imageBoard2.querySelectorAll('.dot');
    existingDots2.forEach(dot => imageBoard2.removeChild(dot));
    const existingCross2 = imageBoard2.querySelectorAll('.cross');
    existingCross2.forEach(cross => imageBoard2.removeChild(cross));
  })

  // var currentValue = slider.value;
  // console.log('@@board-left', clickableImage1.offsetLeft);
  const padX = clickableImage1.offsetLeft;
  const padY = clickableImage1.offsetLeft;
  const dotRadius = 6;
  const crossRadius = 6;
  // console.log('@@padX', padX);
  // console.log('@@padY', padY);
  let RandColor = getRandomColor();
  initialDot.style.backgroundColor = RandColor;
  clickableImage1.addEventListener('mousemove', function (event) {
    initialDot.style.left = `${event.offsetX + padX - dotRadius}px`;
    initialDot.style.top = `${event.offsetY + padY - dotRadius}px`;
  })

  function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    // exclude #000
    if (color === '#000000') {
      return getRandomColor();
    }
    return color;
  }

  function calculateNaturalLocation(x, y) {
    const scaleX = clickableImage1.naturalWidth / clickableImage1.width;
    const scaleY = clickableImage1.naturalHeight / clickableImage1.height;
    const naturalX = Math.round(x * scaleX);
    const naturalY = Math.round(y * scaleY);
    return { x: naturalX, y: naturalY };
  }

  function calculateViewLocation(x, y) {
    const scaleX = clickableImage2.width / clickableImage2.naturalWidth;
    const scaleY = clickableImage2.height / clickableImage2.naturalHeight;
    const viewX = Math.round(x * scaleX);
    const viewY = Math.round(y * scaleY);
    return { x: viewX, y: viewY };
  }


  function transferDots() {
    var imageSrc = "static/demo/flows/" + String(slider.value).padStart(5, '0') + ".png";
    var maskSrc = "static/demo/flows/" + String(slider.value).padStart(5, '0') + "_occ.png";

    var image = images.flow[slider.value];
    var mask = images.mask[slider.value];

    var image_canvas = document.createElement('canvas');
    image_canvas.width = image.width;
    image_canvas.height = image.height;
    var image_ctx = image_canvas.getContext('2d', { willReadFrequently: true });
    image_ctx.drawImage(image, 0, 0);

    var mask_canvas = document.createElement('canvas');
    mask_canvas.width = mask.width;
    mask_canvas.height = mask.height;
    var mask_ctx = mask_canvas.getContext('2d', { willReadFrequently: true });
    mask_ctx.drawImage(mask, 0, 0);

    function read_pixel(ctx, x, y) {
      var pixelData = ctx.getImageData(x, y, 1, 1).data;
      var red = new Int32Array([pixelData[0]])[0];
      var green = new Int32Array([pixelData[1]])[0];
      var blue = new Int32Array([pixelData[2]])[0];
      var alpha = new Int32Array([pixelData[3]])[0];
      return {
        red: red,
        green: green,
        blue: blue,
        alpha: alpha
      };
    }
    // Remove old dots.
    for (olddot of imageBoard2.querySelectorAll('div')) {
      olddot.remove();
    }
    // Add new ones.
    imageBoard1.querySelectorAll('.dot').forEach(dot => {
      const x = dot.natural.x;
      const y = dot.natural.y;
      const color = dot.style.backgroundColor;

      var { red, green, blue, alpha } = read_pixel(image_ctx, x, y);
      flow_x = ((red << 4) | (green >> 4)) - 2048;
      flow_y = ((blue << 4) | (green & 0b1111)) - 2048;
      
      const loc2 = calculateViewLocation(x + flow_x, y + flow_y);
      if (loc2.x > clickableImage2.width || loc2.y > clickableImage2.height
          || loc2.x < 0 || loc2.y < 0
      ) {
        // Out of bounds, don't show this dot.
        return;
      }
      var { red, green, blue, alpha } = read_pixel(mask_ctx, x, y);
      const visible = (red == 0);
      const dot2 = document.createElement('div');
      dot2.style.backgroundColor = color;
      if (visible) {
        dot2.classList.add('dot');
        dot2.style.left = `${loc2.x + padX - dotRadius}px`;
        dot2.style.top = `${loc2.y + padY - dotRadius}px`;
      } else {
        dot2.classList.add('cross');
        dot2.style.left = `${loc2.x + padX - crossRadius}px`;
        dot2.style.top = `${loc2.y + padY - crossRadius}px`;
      }
      imageBoard2.appendChild(dot2);
    });
  }

  clickableImage1.addEventListener('click', function (event) {
    // console.log("@before input: original dot location", event.offsetX, event.offsetY);
    // dot coordinate
    const offsetX = event.offsetX + padX - dotRadius;
    const offsetY = event.offsetY + padY - dotRadius;
    // randColor = getRandomColor();
    const dot1 = document.createElement('div');
    dot1.classList.add('dot');
    dot1.style.backgroundColor = RandColor;
    // dot1.color = RandColor;
    dot1.style.left = `${offsetX}px`;
    dot1.style.top = `${offsetY}px`;
    dot1.natural = calculateNaturalLocation(event.offsetX, event.offsetY);
    imageBoard1.appendChild(dot1);
    // update color
    RandColor = getRandomColor();
    initialDot.style.backgroundColor = RandColor;
    // console.log('Dot Location:', { offsetX, offsetY });

    // the coordinate of pointer in img
    const p_x = event.offsetX;
    const p_y = event.offsetY;
    
    // console.log('@@Visual Img Point Location:', { p_x, p_y });
    console.log('@@color', RandColor);
    transferDots();

    function getCurrentSliderValue(sliderId) {
      const slider = document.getElementById(sliderId);
      const sliderValue = slider.value;
      return sliderValue;
    }
    const sliderValue = getCurrentSliderValue('slider');
    console.log('Slider value:', sliderValue);
  });

  slider.addEventListener('input', transferDots);
});
